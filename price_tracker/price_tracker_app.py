import os
import re
import smtplib
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Optional, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression


app = Flask(__name__)
DB_PATH = "price_tracker.db"
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            category TEXT,
            alert_email TEXT NOT NULL,
            target_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    existing_columns = {
        row["name"] for row in cursor.execute("PRAGMA table_info(products)").fetchall()
    }
    if "alert_email" not in existing_columns:
        cursor.execute("ALTER TABLE products ADD COLUMN alert_email TEXT")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            price REAL NOT NULL,
            currency TEXT DEFAULT 'INR',
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT NOT NULL,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


def extract_numeric_price(raw_text: str) -> Optional[float]:
    if not raw_text:
        return None
    cleaned = re.sub(r"[^\d.]", "", raw_text.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def scrape_amazon_price(url: str) -> Tuple[Optional[str], Optional[float], str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_selectors = ["#productTitle", "span#title", "h1.a-size-large"]
    price_selectors = [
        "span.a-price > span.a-offscreen",
        "span.a-price-whole",
        "span#priceblock_ourprice",
        "span#priceblock_dealprice",
        "span#priceblock_saleprice",
    ]

    title = None
    for selector in title_selectors:
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break

    price = None
    for selector in price_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        parsed = extract_numeric_price(node.get_text(strip=True))
        if parsed:
            price = parsed
            break

    if not title:
        title = "Amazon Product"

    if price is None:
        raise ValueError("Amazon price element not found. The selector may need updating.")

    return title, price, "INR"


def scrape_flipkart_price(url: str) -> Tuple[Optional[str], Optional[float], str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_selectors = ["span.B_NuCI", "h1._6EBuvT span", "h1.yhB1nd span"]
    price_selectors = [
        "div._30jeq3._16Jk6d",
        "div.Nx9bqj.CxhGGd",
        "div._30jeq3",
        "div._1_WHN1",
    ]

    title = None
    for selector in title_selectors:
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break

    price = None
    for selector in price_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        parsed = extract_numeric_price(node.get_text(strip=True))
        if parsed:
            price = parsed
            break

    if not title:
        title = "Flipkart Product"

    if price is None:
        raise ValueError("Flipkart price element not found. The selector may need updating.")

    return title, price, "INR"


def scrape_product(url: str) -> Tuple[str, float, str]:
    lowered = url.lower()
    if "amazon." in lowered:
        return scrape_amazon_price(url)
    if "flipkart." in lowered:
        return scrape_flipkart_price(url)
    raise ValueError("Unsupported URL. Only Amazon and Flipkart links are allowed.")


def add_price_entry(product_id: int, price: float, currency: str = "INR") -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO price_history (product_id, price, currency) VALUES (?, ?, ?)",
        (product_id, price, currency),
    )
    conn.commit()
    conn.close()


def latest_price(product_id: int) -> Optional[float]:
    conn = get_db_connection()
    row = conn.execute(
        "SELECT price FROM price_history WHERE product_id = ? ORDER BY scraped_at DESC LIMIT 1",
        (product_id,),
    ).fetchone()
    conn.close()
    return float(row["price"]) if row else None


def create_alert(product_id: int, alert_type: str, message: str) -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO alerts (product_id, alert_type, message) VALUES (?, ?, ?)",
        (product_id, alert_type, message),
    )
    conn.commit()
    conn.close()


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_REGEX.match(value))


def send_alert_email(recipient: str, subject: str, body: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASS")
    sender_email = os.environ.get("ALERT_SENDER_EMAIL") or smtp_user
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"

    if not all([smtp_host, smtp_user, smtp_password, sender_email]):
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
        if use_tls:
            smtp.starttls()
        smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)
    return True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/products", methods=["GET"])
def get_products():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT p.id, p.product_name, p.url, p.category, p.target_price, p.created_at,
               p.alert_email,
               ph.price AS latest_price, ph.scraped_at AS latest_scraped_at
        FROM products p
        LEFT JOIN price_history ph ON ph.id = (
            SELECT id FROM price_history
            WHERE product_id = p.id
            ORDER BY scraped_at DESC
            LIMIT 1
        )
        ORDER BY p.created_at DESC
        """
    ).fetchall()
    conn.close()

    products = [dict(row) for row in rows]
    return jsonify({"products": products})


@app.route("/api/products/add", methods=["POST"])
def add_product():
    payload = request.get_json(silent=True) or {}
    url = (payload.get("url") or "").strip()
    category = (payload.get("category") or "General").strip()
    alert_email = (payload.get("alert_email") or "").strip().lower()
    target_price = payload.get("target_price")

    if not url:
        return jsonify({"error": "Product URL is required."}), 400
    if not alert_email:
        return jsonify({"error": "Alert email is required."}), 400
    if not is_valid_email(alert_email):
        return jsonify({"error": "Please enter a valid email address."}), 400

    try:
        product_name, price, currency = scrape_product(url)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO products (product_name, url, category, alert_email, target_price)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                product_name,
                url,
                category,
                alert_email,
                float(target_price) if target_price not in (None, "", "null") else None,
            ),
        )
        product_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO price_history (product_id, price, currency) VALUES (?, ?, ?)",
            (product_id, price, currency),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "This product URL is already being tracked."}), 409
    finally:
        conn.close()

    return jsonify(
        {
            "message": "Product added successfully.",
            "product": {
                "id": product_id,
                "product_name": product_name,
                "url": url,
                "category": category,
                "alert_email": alert_email,
                "target_price": target_price,
                "latest_price": price,
            },
        }
    )


@app.route("/api/products/<int:product_id>/delete", methods=["DELETE"])
def delete_product(product_id: int):
    conn = get_db_connection()
    result = conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
    conn.commit()
    conn.close()
    if result.rowcount == 0:
        return jsonify({"error": "Product not found."}), 404
    return jsonify({"message": "Product deleted successfully."})


@app.route("/api/products/<int:product_id>/history", methods=["GET"])
def product_history(product_id: int):
    days = request.args.get("days", default=30, type=int)
    since_dt = datetime.utcnow() - timedelta(days=max(days, 1))
    since_text = since_dt.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT price, currency, scraped_at
        FROM price_history
        WHERE product_id = ? AND scraped_at >= ?
        ORDER BY scraped_at ASC
        """,
        (product_id, since_text),
    ).fetchall()
    conn.close()

    history = [dict(row) for row in rows]
    return jsonify({"history": history})


@app.route("/api/products/<int:product_id>/refresh", methods=["POST"])
def refresh_product(product_id: int):
    conn = get_db_connection()
    product = conn.execute(
        "SELECT id, url, target_price, product_name, alert_email FROM products WHERE id = ?",
        (product_id,),
    ).fetchone()
    conn.close()

    if not product:
        return jsonify({"error": "Product not found."}), 404

    prev_price = latest_price(product_id)
    try:
        _, new_price, currency = scrape_product(product["url"])
        add_price_entry(product_id, new_price, currency)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    alerts = []
    mail_results = []
    if prev_price:
        drop_pct = ((prev_price - new_price) / prev_price) * 100
        if drop_pct > 5:
            msg = f"Price dropped by {drop_pct:.2f}% for {product['product_name']}."
            create_alert(product_id, "price_drop", msg)
            alerts.append(msg)
            if product["alert_email"]:
                sent = send_alert_email(
                    product["alert_email"],
                    f"Price Drop Alert: {product['product_name']}",
                    f"{msg}\n\nPrevious: Rs {prev_price:.2f}\nCurrent: Rs {new_price:.2f}\nURL: {product['url']}",
                )
                mail_results.append({"type": "price_drop", "sent": sent})

    target_price = product["target_price"]
    if target_price is not None and new_price <= float(target_price):
        msg = f"Target reached for {product['product_name']}: Rs {new_price:.2f}."
        create_alert(product_id, "target_reached", msg)
        alerts.append(msg)
        if product["alert_email"]:
            sent = send_alert_email(
                product["alert_email"],
                f"Target Reached: {product['product_name']}",
                f"{msg}\n\nTarget: Rs {float(target_price):.2f}\nCurrent: Rs {new_price:.2f}\nURL: {product['url']}",
            )
            mail_results.append({"type": "target_reached", "sent": sent})

    return jsonify(
        {
            "message": "Price refreshed successfully.",
            "latest_price": new_price,
            "previous_price": prev_price,
            "alerts": alerts,
            "email_notifications": mail_results,
        }
    )


@app.route("/api/products/<int:product_id>/predict", methods=["GET"])
def predict_price(product_id: int):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT price, scraped_at
        FROM price_history
        WHERE product_id = ?
        ORDER BY scraped_at ASC
        LIMIT 60
        """,
        (product_id,),
    ).fetchall()
    conn.close()

    if len(rows) < 5:
        return jsonify({"error": "Need at least 5 price records for prediction."}), 400

    prices = np.array([float(r["price"]) for r in rows])
    x = np.arange(len(prices)).reshape(-1, 1)
    y = prices.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    future_day = np.array([[len(prices) + 7]])
    predicted_price = float(model.predict(future_day)[0][0])
    current_price = float(prices[-1])
    drop_pct = ((current_price - predicted_price) / current_price) * 100 if current_price else 0.0

    return jsonify(
        {
            "current_price": current_price,
            "predicted_price_7d": round(predicted_price, 2),
            "predicted_drop_percent": round(drop_pct, 2),
            "likely_to_drop": drop_pct > 2,
        }
    )


@app.route("/api/stats", methods=["GET"])
def stats():
    conn = get_db_connection()
    product_count = conn.execute("SELECT COUNT(*) AS c FROM products").fetchone()["c"]
    price_count = conn.execute("SELECT COUNT(*) AS c FROM price_history").fetchone()["c"]

    rows = conn.execute(
        """
        SELECT product_id, MIN(price) AS min_p, MAX(price) AS max_p
        FROM price_history
        GROUP BY product_id
        """
    ).fetchall()
    conn.close()

    variations = []
    for row in rows:
        min_p = float(row["min_p"])
        max_p = float(row["max_p"])
        if max_p > 0:
            variations.append(((max_p - min_p) / max_p) * 100)
    avg_variation = round(sum(variations) / len(variations), 2) if variations else 0.0

    return jsonify(
        {
            "total_products": product_count,
            "price_records": price_count,
            "average_variation_percent": avg_variation,
        }
    )


def background_scraper(interval_seconds: int = 3600) -> None:
    while True:
        conn = get_db_connection()
        products = conn.execute("SELECT id FROM products").fetchall()
        conn.close()

        for product in products:
            try:
                refresh_product(product["id"])
            except Exception:
                # Non-fatal; scraper continues with other products.
                pass

        time.sleep(interval_seconds)


if __name__ == "__main__":
    init_db()
    # Automatic hourly refresh for all tracked products.
    scraper_thread = threading.Thread(target=background_scraper, daemon=True)
    scraper_thread.start()
    app.run(debug=True, port=5000)
