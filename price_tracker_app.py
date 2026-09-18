import os
import logging
import json
import re
import smtplib
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Optional, Tuple

# Reduce noisy OpenBLAS thread allocation on some Windows setups.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        # Always prefer values from `price_tracker/.env` over inherited shell env vars.
        load_dotenv(dotenv_path=env_path, override=True)
    except Exception:
        # Keep startup resilient even if python-dotenv is unavailable.
        pass


load_env_file()


app = Flask(__name__)
DB_PATH = "price_tracker.db"


@app.after_request
def add_cors_headers(response):
    # Lets the UI work if someone opens the HTML file directly while the server is running.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


@app.before_request
def handle_cors_preflight():
    if request.method == "OPTIONS":
        return app.make_response(("", 204))
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("price_tracker.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("price-tracker")

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
            email_to TEXT,
            email_sent INTEGER DEFAULT 0,
            email_error TEXT,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        """
    )
    alert_columns = {
        row["name"] for row in cursor.execute("PRAGMA table_info(alerts)").fetchall()
    }
    if "email_to" not in alert_columns:
        cursor.execute("ALTER TABLE alerts ADD COLUMN email_to TEXT")
    if "email_sent" not in alert_columns:
        cursor.execute("ALTER TABLE alerts ADD COLUMN email_sent INTEGER DEFAULT 0")
    if "email_error" not in alert_columns:
        cursor.execute("ALTER TABLE alerts ADD COLUMN email_error TEXT")

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


def _cffi_session():
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError as exc:
        raise ValueError(
            "Price scraping requires curl_cffi. Install it with: pip install curl_cffi"
        ) from exc

    return cffi_requests.Session()


def _is_amazon_block_page(html: str) -> bool:
    lowered = html.lower()
    return (
        "validatecaptcha" in lowered
        or "opfcaptcha" in lowered
        or "enter the characters you see below" in lowered
        or ("sorry" in lowered and "robot" in lowered and len(html) < 20000)
    )


def fetch_amazon_html(url: str) -> str:
    session = _cffi_session()
    headers = {
        "Referer": "https://www.amazon.in/",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        session.get("https://www.amazon.in/", impersonate="chrome", timeout=20, headers=headers)
    except Exception:
        logger.debug("Amazon homepage warmup failed; continuing with product fetch.")

    response = session.get(url, impersonate="chrome", timeout=25, headers=headers)
    response.raise_for_status()
    html = response.text

    if _is_amazon_block_page(html) or ("id=\"productTitle\"" not in html and len(html) < 20000):
        raise ValueError(
            "Amazon blocked the request (captcha/bot protection). Please retry after a minute."
        )
    return html


def fetch_flipkart_html(url: str) -> str:
    session = _cffi_session()
    session.get("https://www.flipkart.com/", impersonate="chrome", timeout=20)
    response = session.get(
        url,
        impersonate="chrome",
        timeout=20,
        headers={"Referer": "https://www.flipkart.com/"},
    )
    response.raise_for_status()

    if "reCAPTCHA" in response.text or len(response.text) < 5000:
        raise ValueError(
            "Flipkart blocked the request (captcha/bot protection). Please retry after a minute."
        )
    return response.text


def parse_flipkart_initial_state(html: str) -> Optional[dict]:
    marker = "window.__INITIAL_STATE__ = "
    start = html.find(marker)
    if start < 0:
        return None
    json_start = start + len(marker)
    try:
        state, _ = json.JSONDecoder().raw_decode(html[json_start:])
        return state if isinstance(state, dict) else None
    except json.JSONDecodeError:
        return None


def extract_flipkart_price(state: Optional[dict], soup: BeautifulSoup, html: str) -> Optional[float]:
    if state:
        try:
            ppd = (
                state["multiWidgetState"]["pageDataResponse"]["pageContext"]["fdpEventTracking"]["events"]["psi"]["ppd"]
            )
            for key in ("finalPrice", "fsp", "mrp"):
                value = ppd.get(key)
                if value is not None:
                    return float(value)
        except (KeyError, TypeError, ValueError):
            pass

        def walk(obj):
            if isinstance(obj, dict):
                if "finalPrice" in obj and isinstance(obj["finalPrice"], (int, float)):
                    return float(obj["finalPrice"])
                if "fsp" in obj and isinstance(obj["fsp"], (int, float)):
                    return float(obj["fsp"])
                for value in obj.values():
                    found = walk(value)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = walk(item)
                    if found:
                        return found
            return None

        walked = walk(state)
        if walked:
            return walked

    price_selectors = [
        "div._30jeq3._16Jk6d",
        "div.Nx9bqj.CxhGGd",
        "div._30jeq3",
        "div._1_WHN1",
        "div[class*='Nx9bqj']",
        "div[class*='30jeq3']",
    ]
    for selector in price_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        parsed = extract_numeric_price(node.get_text(strip=True))
        if parsed:
            return parsed

    rupee_prices = []
    for match in re.findall(r"\u20b9\s*([\d,]+)", html):
        parsed = extract_numeric_price(match)
        if parsed and parsed >= 100:
            rupee_prices.append(parsed)
    if rupee_prices:
        return max(rupee_prices)
    return None


def extract_flipkart_title(soup: BeautifulSoup) -> str:
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    for selector in ["span.B_NuCI", "h1._6EBuvT span", "h1.yhB1nd span", "h1"]:
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            return node.get_text(strip=True)

    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Flipkart Product"


def extract_amazon_title(soup: BeautifulSoup) -> str:
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    for selector in ["#productTitle", "span#title", "h1.a-size-large", "h1"]:
        node = soup.select_one(selector)
        if node and node.get_text(strip=True):
            return node.get_text(strip=True)

    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Amazon Product"


def extract_amazon_price(soup: BeautifulSoup, html: str) -> Optional[float]:
    twister = soup.select_one("#twister-plus-price-data-price, input[name='items[0.base][customerVisiblePrice][amount]']")
    if twister:
        parsed = extract_numeric_price(twister.get("value") or twister.get_text(strip=True))
        if parsed:
            return parsed

    price_selectors = [
        "#corePriceDisplay_desktop_feature_div span.a-price.aok-align-center span.a-offscreen",
        "#corePrice_feature_div span.a-offscreen",
        "#apex_desktop span.a-price.aok-align-center span.a-offscreen",
        "#priceToPay span.a-offscreen",
        "span.a-price.apex-pricetopay-value > span.a-offscreen",
        "#corePriceDisplay_desktop_feature_div span.a-price-whole",
        "span.a-price > span.a-offscreen",
        "span#priceblock_ourprice",
        "span#priceblock_dealprice",
        "span#priceblock_saleprice",
        "span.a-price-whole",
    ]
    for selector in price_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        parsed = extract_numeric_price(node.get_text(strip=True))
        if parsed:
            return parsed

    for script in soup.select('script[type="application/ld+json"]'):
        raw = (script.string or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            offers = item.get("offers")
            offer_list = offers if isinstance(offers, list) else [offers]
            for offer in offer_list:
                if not isinstance(offer, dict):
                    continue
                parsed = extract_numeric_price(str(offer.get("price") or ""))
                if parsed:
                    return parsed

    amount = re.search(r'"priceAmount"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', html)
    if amount:
        parsed = extract_numeric_price(amount.group(1))
        if parsed:
            return parsed
    return None


def scrape_amazon_price(url: str) -> Tuple[Optional[str], Optional[float], str]:
    html = fetch_amazon_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title = extract_amazon_title(soup)
    price = extract_amazon_price(soup, html)

    if price is None:
        raise ValueError("Amazon price element not found. The selector may need updating.")

    return title, price, "INR"


def scrape_flipkart_price(url: str) -> Tuple[Optional[str], Optional[float], str]:
    html = fetch_flipkart_html(url)
    soup = BeautifulSoup(html, "html.parser")
    state = parse_flipkart_initial_state(html)

    title = extract_flipkart_title(soup)
    price = extract_flipkart_price(state, soup, html)

    if price is None:
        raise ValueError(
            "Flipkart price not found. Flipkart may have changed its page structure or blocked the request."
        )

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


def create_alert(
    product_id: int,
    alert_type: str,
    message: str,
    email_to: Optional[str] = None,
    email_sent: bool = False,
    email_error: Optional[str] = None,
) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO alerts (product_id, alert_type, message, email_to, email_sent, email_error)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (product_id, alert_type, message, email_to, 1 if email_sent else 0, email_error),
    )
    conn.commit()
    conn.close()


def is_valid_email(value: str) -> bool:
    return bool(EMAIL_REGEX.match(value))


def send_alert_email(recipient: str, subject: str, body: str) -> None:
    smtp_host = (os.environ.get("SMTP_HOST") or "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = (os.environ.get("SMTP_USER") or "").strip()
    smtp_password_raw = os.environ.get("SMTP_PASS")
    smtp_password = (smtp_password_raw or "").strip().strip('"').strip("'")
    sender_email = (os.environ.get("ALERT_SENDER_EMAIL") or smtp_user or "").strip()
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"
    use_ssl = os.environ.get("SMTP_USE_SSL", "false").lower() == "true"

    if not all([smtp_host, smtp_user, smtp_password, sender_email]):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        raise RuntimeError(
            "SMTP is not configured. Create/update `price_tracker/.env` with "
            "SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS (and optionally ALERT_SENDER_EMAIL). "
            f"Expected file: {env_path}"
        )

    placeholder_user = "yourgmail@gmail.com"
    placeholder_pass = "your_16_char_app_password"
    if smtp_user.lower() == placeholder_user or smtp_password.lower() == placeholder_pass:
        raise RuntimeError(
            "Your `price_tracker/.env` still contains template placeholder SMTP values. "
            f"Update SMTP_USER (currently `{smtp_user}`) and SMTP_PASS (still the template password) "
            "to your real Gmail address + Google App Password, then restart the server."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient
    msg.set_content(body)

    try:
        if use_ssl or smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as smtp:
                smtp.login(smtp_user, smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
                if use_tls:
                    smtp.starttls()
                smtp.login(smtp_user, smtp_password)
                smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            "Gmail rejected SMTP login (535 BadCredentials). "
            "Fix checklist: enable 2-Step Verification, create an App Password, "
            "set SMTP_USER to the exact Gmail address, set SMTP_PASS to the App Password (not your normal password), "
            "and set ALERT_SENDER_EMAIL to the same Gmail address. "
            "If you still see 535, rotate the App Password and try again."
        ) from exc


def refresh_product_core(product_id: int) -> dict:
    conn = get_db_connection()
    product = conn.execute(
        "SELECT id, url, target_price, product_name, alert_email FROM products WHERE id = ?",
        (product_id,),
    ).fetchone()
    conn.close()

    if not product:
        raise ValueError("Product not found.")

    prev_price = latest_price(product_id)
    _, new_price, currency = scrape_product(product["url"])
    add_price_entry(product_id, new_price, currency)

    alerts = []
    mail_results = []

    def try_send(email_to: str, subject: str, body: str, alert_type: str, message: str) -> None:
        sent = False
        err = None
        try:
            send_alert_email(email_to, subject, body)
            sent = True
            logger.info("Email sent: product_id=%s type=%s to=%s", product_id, alert_type, email_to)
        except Exception as exc:
            err = str(exc)
            logger.exception("Email failed: product_id=%s type=%s to=%s", product_id, alert_type, email_to)
        create_alert(product_id, alert_type, message, email_to=email_to, email_sent=sent, email_error=err)
        mail_results.append({"type": alert_type, "to": email_to, "sent": sent, "error": err})

    if prev_price:
        drop_pct = ((prev_price - new_price) / prev_price) * 100
        if drop_pct > 5:
            msg = f"Price dropped by {drop_pct:.2f}% for {product['product_name']}."
            alerts.append(msg)
            if product["alert_email"]:
                try_send(
                    product["alert_email"],
                    f"Price Drop Alert: {product['product_name']}",
                    f"{msg}\n\nPrevious: Rs {prev_price:.2f}\nCurrent: Rs {new_price:.2f}\nURL: {product['url']}",
                    "price_drop",
                    msg,
                )
            else:
                create_alert(product_id, "price_drop", msg)

    target_price = product["target_price"]
    if target_price is not None and new_price <= float(target_price):
        msg = f"Target reached for {product['product_name']}: Rs {new_price:.2f}."
        alerts.append(msg)
        if product["alert_email"]:
            try_send(
                product["alert_email"],
                f"Target Reached: {product['product_name']}",
                f"{msg}\n\nTarget: Rs {float(target_price):.2f}\nCurrent: Rs {new_price:.2f}\nURL: {product['url']}",
                "target_reached",
                msg,
            )
        else:
            create_alert(product_id, "target_reached", msg)

    return {
        "latest_price": new_price,
        "previous_price": prev_price,
        "alerts": alerts,
        "email_notifications": mail_results,
    }


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
    try:
        result = refresh_product_core(product_id)
        return jsonify({"message": "Price refreshed successfully.", **result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Refresh failed: product_id=%s", product_id)
        return jsonify({"error": str(exc)}), 400


@app.route("/api/test-email", methods=["POST"])
def test_email():
    payload = request.get_json(silent=True) or {}
    recipient = (payload.get("email") or "").strip().lower()

    if not recipient:
        return jsonify({"error": "Email is required."}), 400
    if not is_valid_email(recipient):
        return jsonify({"error": "Please enter a valid email address."}), 400

    subject = "Smart Price Tracker - Test Email"
    body = (
        "This is a test email from Smart Price Tracker.\n\n"
        "If you received this message, your SMTP configuration is working correctly."
    )
    try:
        send_alert_email(recipient, subject, body)
        logger.info("Test email sent to %s", recipient)
        return jsonify({"message": f"Test email sent to {recipient}."})
    except Exception as exc:
        logger.exception("Test email failed to %s", recipient)
        return jsonify({"error": f"Failed to send test email: {exc}"}), 400


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
                refresh_product_core(product["id"])
            except Exception:
                logger.exception("Background refresh failed: product_id=%s", product["id"])

        time.sleep(interval_seconds)


if __name__ == "__main__":
    init_db()
    # Automatic hourly refresh for all tracked products.
    scraper_thread = threading.Thread(target=background_scraper, daemon=True)
    scraper_thread.start()
    app.run(debug=True, port=5000)
