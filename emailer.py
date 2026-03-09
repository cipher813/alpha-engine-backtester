"""
emailer.py — send weekly backtest report via AWS SES.

Matches the style and config conventions of executor/eod_emailer.py.
Sender and recipients come from config.yaml (email_sender / email_recipients).
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_HTML = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body  {{ font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6;
           color: #222; max-width: 720px; margin: 0 auto; padding: 20px; }}
  h1   {{ font-size: 16px; border-bottom: 2px solid #555; padding-bottom: 6px; }}
  h2   {{ font-size: 14px; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  th   {{ background: #f0f0f0; }}
  blockquote {{ margin: 4px 0 4px 12px; color: #666; border-left: 3px solid #ccc; padding-left: 8px; }}
  pre  {{ background: #f8f8f8; padding: 10px; font-size: 12px; overflow-x: auto; }}
  .foot {{ margin-top: 28px; font-size: 11px; color: #888;
            border-top: 1px solid #ccc; padding-top: 8px; }}
</style>
</head>
<body>
{body}
<div class="foot">Alpha Engine Backtester | {date}{s3_link}</div>
</body>
</html>
"""


def send_report_email(
    run_date: str,
    report_md: str,
    status: str,
    sender: str,
    recipients: list[str],
    s3_bucket: str | None = None,
    s3_prefix: str = "backtest",
    region: str = "us-east-1",
) -> None:
    """
    Send the weekly backtest report via SES.

    Args:
        run_date:    Date string for subject line and footer.
        report_md:   Markdown report string from reporter.build_report().
        status:      "ok" | "insufficient_data" | "error" — shown in subject.
        sender:      SES-verified from address.
        recipients:  List of to addresses.
        s3_bucket:   If set, include S3 link to report in footer.
        s3_prefix:   S3 prefix for report location (default "backtest").
        region:      AWS region for SES.
    """
    subject = _build_subject(run_date, status)
    html_body, plain_body = _build_body(run_date, report_md, s3_bucket, s3_prefix)

    ses = boto3.client("ses", region_name=region)
    try:
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        logger.info("Backtest report email sent: '%s' → %s", subject, recipients)
    except ClientError as e:
        logger.error("SES send failed: %s", e.response["Error"]["Message"])
    except Exception as e:
        logger.error("Email error: %s", e)


def _build_subject(run_date: str, status: str) -> str:
    label = {
        "ok": "results ready",
        "insufficient_data": "insufficient data (accumulating)",
        "db_not_found": "ERROR — research.db not found",
        "error": "ERROR",
    }.get(status, status)
    return f"Alpha Engine Backtester | {run_date} | {label}"


def _build_body(
    run_date: str,
    report_md: str,
    s3_bucket: str | None,
    s3_prefix: str,
) -> tuple[str, str]:
    # Convert minimal markdown to HTML (tables, headers, blockquotes, hr)
    html_lines = []
    for line in report_md.splitlines():
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        elif line.startswith("---"):
            html_lines.append("<hr>")
        elif line.startswith("|"):
            html_lines.append(_md_table_row(line))
        elif line.startswith("_") and line.endswith("_"):
            html_lines.append(f"<p><em>{line.strip('_')}</em></p>")
        elif line.strip() == "":
            html_lines.append("")
        else:
            html_lines.append(f"<p>{line}</p>")

    s3_link = ""
    if s3_bucket:
        url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_prefix}/{run_date}/report.md"
        s3_link = f' | <a href="{url}">S3 report</a>'

    html_body = _HTML.format(
        body="\n".join(html_lines),
        date=run_date,
        s3_link=s3_link,
    )
    return html_body, report_md  # plain body is just the markdown


def _md_table_row(line: str) -> str:
    """Convert a markdown table row to an HTML table row."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(set(c) <= set("-: ") for c in cells):
        return ""  # separator row
    tag = "th" if any(c.strip().startswith("**") or line.startswith("| Metric") or
                      line.startswith("| File") or line.startswith("| Bucket") or
                      line.startswith("| Min") or line.startswith("| Regime") or
                      line.startswith("| Sub") or line.startswith("| Threshold")
                      for c in cells) else "td"
    inner = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
    return f"<tr>{inner}</tr>"
