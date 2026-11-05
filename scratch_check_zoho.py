import os
from dotenv import load_dotenv
from zoho_crm import _get_access_token, _get_config
import urllib.request
import json

load_dotenv()
config = _get_config()
token = _get_access_token()
print("Token obtained successfully.")

url = f"{config['api_domain']}/crm/v2/{config['module']}?fields=Full_Name,Company,Email,Phone,Lead_Source"
req = urllib.request.Request(
    url,
    headers={
        "Authorization": f"Zoho-oauthtoken {token}",
    },
    method="GET"
)

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        leads = data.get("data", [])
        print(f"Found {len(leads)} leads in Zoho CRM:")
        for l in leads:
            print("ID:", l.get("id"), "| Name:", l.get("Full_Name"), "| Company:", l.get("Company"), "| Email:", l.get("Email"), "| Source:", l.get("Lead_Source"))
except urllib.error.HTTPError as e:
    print("HTTP error:", e.code, e.read().decode())

