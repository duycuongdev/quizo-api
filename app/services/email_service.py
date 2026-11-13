"""Transactional email service for Quizo via Brevo SMTP / Dev logger."""

import asyncio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import smtplib

from app.core.config import settings

logger = logging.getLogger("quizo.email")


def _send_smtp_sync(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    """Synchronous SMTP sender run inside asyncio.to_thread."""
    if not settings.SMTP_SERVER or not settings.SMTP_PASSWORD:
        logger.info(
            "[DEV EMAIL] No SMTP credentials configured. Email to <%s> logged:\nSubject: %s\n%s",
            to_email,
            subject,
            text_body,
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    from_addr = settings.SMTP_FROM_EMAIL or settings.SMTP_USER or "noreply@quizo.app"
    from_name = settings.SMTP_FROM_NAME or "Quizo"
    msg["From"] = f"{from_name} <{from_addr}>"
    msg["To"] = to_email

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        logger.info("Successfully sent email to %s via %s", to_email, settings.SMTP_SERVER)
        return True
    except Exception as exc:
        logger.warning(
            "Failed to send email to %s via Brevo SMTP (%s: %s). "
            "Logging action link for local testing:\n%s",
            to_email,
            type(exc).__name__,
            exc,
            text_body,
        )
        return False


async def send_verification_email(email: str, full_name: str, token: str) -> None:
    """Send an account verification email containing an activation link."""
    verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    subject = "Quizo — Xác nhận địa chỉ email của bạn"

    text_body = f"""Xin chào {full_name},

Cảm ơn bạn đã đăng ký tài khoản trên Quizo — Nền tảng học AI Engineer qua Trắc nghiệm & AI Studio.

Vui lòng nhấn vào đường link dưới đây để xác nhận địa chỉ email của bạn (liên kết có hiệu lực trong {settings.EMAIL_VERIFICATION_EXPIRE_HOURS} giờ):
{verify_url}

Nếu bạn không thực hiện đăng ký tài khoản này, vui lòng bỏ qua email này.

Trân trọng,
Đội ngũ Quizo
"""

    html_body = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin:0; padding:0; background-color:#F5F7FF; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#1E1B4B; }}
    .container {{ max-width:560px; margin:40px auto; background:#FFFFFF; border-radius:24px; padding:36px 32px; border:1px solid #D8DCF0; box-shadow:0 10px 30px rgba(79,70,229,0.06); }}
    .logo {{ display:inline-block; font-size:24px; font-weight:800; color:#4F46E5; text-decoration:none; margin-bottom:20px; }}
    .title {{ font-size:22px; font-weight:700; color:#1E1B4B; margin:0 0 14px 0; }}
    .desc {{ font-size:15px; line-height:1.6; color:#475569; margin:0 0 24px 0; }}
    .btn-wrap {{ text-align:center; margin:30px 0; }}
    .btn {{ display:inline-block; background-color:#4F46E5; color:#FFFFFF !important; text-decoration:none; font-weight:600; font-size:15px; padding:14px 32px; border-radius:14px; box-shadow:0 4px 14px rgba(79,70,229,0.3); }}
    .footer {{ margin-top:32px; padding-top:20px; border-top:1px solid #E2E8F0; font-size:12px; color:#94A3B8; line-height:1.5; }}
    .link-fallback {{ word-break:break-all; color:#4F46E5; }}
  </style>
</head>
<body>
  <div class="container">
    <a href="{settings.FRONTEND_URL}" class="logo">⚡ Quizo</a>
    <h1 class="title">Xác nhận email của bạn</h1>
    <p class="desc">
      Xin chào <strong>{full_name}</strong>,<br>
      Chào mừng bạn đến với Quizo! Vui lòng nhấn vào nút bên dưới để kích hoạt tài khoản và mở khóa toàn bộ lộ trình ôn luyện AI Engineer.
    </p>
    <div class="btn-wrap">
      <a href="{verify_url}" class="btn" target="_blank">Xác nhận địa chỉ Email</a>
    </div>
    <p class="desc" style="font-size:13px; color:#64748B;">
      Liên kết xác nhận này có hiệu lực trong <strong>{settings.EMAIL_VERIFICATION_EXPIRE_HOURS} giờ</strong>. Nếu bạn không tạo tài khoản trên Quizo, xin vui lòng bỏ qua thư này.
    </p>
    <div class="footer">
      Nếu nút bấm phía trên không hoạt động, bạn có thể sao chép và dán liên kết sau vào trình duyệt:<br>
      <a href="{verify_url}" class="link-fallback">{verify_url}</a>
    </div>
  </div>
</body>
</html>"""

    await asyncio.to_thread(_send_smtp_sync, email, subject, html_body, text_body)


async def send_password_reset_email(email: str, full_name: str, token: str) -> None:
    """Send a password reset email containing a secure one-time link."""
    reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
    subject = "Quizo — Hướng dẫn đặt lại mật khẩu"

    text_body = f"""Xin chào {full_name},

Chúng tôi nhận được yêu cầu đặt lại mật khẩu cho tài khoản Quizo của bạn.

Vui lòng truy cập đường link sau để tạo mật khẩu mới (liên kết có hiệu lực trong {settings.PASSWORD_RESET_EXPIRE_MINUTES} phút):
{reset_url}

Nếu bạn KHÔNG yêu cầu đặt lại mật khẩu, xin hãy bỏ qua email này. Mật khẩu hiện tại của bạn vẫn được bảo mật an toàn.

Trân trọng,
Đội ngũ Quizo
"""

    html_body = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin:0; padding:0; background-color:#F5F7FF; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:#1E1B4B; }}
    .container {{ max-width:560px; margin:40px auto; background:#FFFFFF; border-radius:24px; padding:36px 32px; border:1px solid #D8DCF0; box-shadow:0 10px 30px rgba(79,70,229,0.06); }}
    .logo {{ display:inline-block; font-size:24px; font-weight:800; color:#4F46E5; text-decoration:none; margin-bottom:20px; }}
    .title {{ font-size:22px; font-weight:700; color:#1E1B4B; margin:0 0 14px 0; }}
    .desc {{ font-size:15px; line-height:1.6; color:#475569; margin:0 0 24px 0; }}
    .btn-wrap {{ text-align:center; margin:30px 0; }}
    .btn {{ display:inline-block; background-color:#4F46E5; color:#FFFFFF !important; text-decoration:none; font-weight:600; font-size:15px; padding:14px 32px; border-radius:14px; box-shadow:0 4px 14px rgba(79,70,229,0.3); }}
    .footer {{ margin-top:32px; padding-top:20px; border-top:1px solid #E2E8F0; font-size:12px; color:#94A3B8; line-height:1.5; }}
    .link-fallback {{ word-break:break-all; color:#4F46E5; }}
  </style>
</head>
<body>
  <div class="container">
    <a href="{settings.FRONTEND_URL}" class="logo">⚡ Quizo</a>
    <h1 class="title">Yêu cầu đặt lại mật khẩu</h1>
    <p class="desc">
      Xin chào <strong>{full_name}</strong>,<br>
      Chúng tôi nhận được yêu cầu đặt lại mật khẩu cho tài khoản Quizo của bạn. Nhấn vào nút bên dưới để thiết lập mật khẩu mới:
    </p>
    <div class="btn-wrap">
      <a href="{reset_url}" class="btn" target="_blank">Đặt lại mật khẩu</a>
    </div>
    <p class="desc" style="font-size:13px; color:#64748B;">
      Liên kết này có hiệu lực trong <strong>{settings.PASSWORD_RESET_EXPIRE_MINUTES} phút</strong>. Nếu bạn không gửi yêu cầu này, vui lòng bỏ qua thư. Tài khoản của bạn vẫn an toàn tuyệt đối.
    </p>
    <div class="footer">
      Nếu nút bấm phía trên không hoạt động, bạn có thể sao chép và dán liên kết sau vào trình duyệt:<br>
      <a href="{reset_url}" class="link-fallback">{reset_url}</a>
    </div>
  </div>
</body>
</html>"""

    await asyncio.to_thread(_send_smtp_sync, email, subject, html_body, text_body)
