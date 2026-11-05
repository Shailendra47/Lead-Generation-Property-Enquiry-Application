import json
import os
import urllib.request
import urllib.error
from dotenv import load_dotenv
from zoho_crm import _get_access_token, _get_config

load_dotenv()
config = _get_config()
target_company = os.getenv("ZOHO_DEFAULT_COMPANY", "Real Estate Company").strip()
token = _get_access_token()

print(f"Target company: '{target_company}'")

# 1. Fetch leads
fetch_url = f"{config['api_domain']}/crm/v2/{config['module']}?fields=Full_Name,Company,Email,Phone"
req = urllib.request.Request(
    fetch_url,
    headers={"Authorization": f"Zoho-oauthtoken {token}"},
    method="GET"
)

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    leads = data.get("data", [])

update_payload = []
for l in leads:
    lead_id = l["id"]
    current_company = l.get("Company")
    name = l.get("Full_Name")
    print(f"Lead {name} (ID: {lead_id}): current Company = '{current_company}'")
    if current_company != target_company:
        update_payload.append({
            "id": lead_id,
            "Company": target_company
        })

if update_payload:
    print(f"\nUpdating {len(update_payload)} leads in Zoho CRM...")
    put_url = f"{config['api_domain']}/crm/v2/{config['module']}"
    req_body = json.dumps({"data": update_payload}).encode("utf-8")
    put_req = urllib.request.Request(
        put_url,
        data=req_body,
        headers={
            "Authorization": f"Zoho-oauthtoken {token}",
            "Content-Type": "application/json"
        },
        method="PUT"
    )
    try:
        with urllib.request.urlopen(put_req) as resp:
            put_resp = json.loads(resp.read().decode())
            print("Zoho CRM Update Response:")
            for row in put_resp.get("data", []):
                print(f"  Status: {row.get('status')}, Code: {row.get('code')}, Details: {row.get('details')}")
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())
else:
    print("All leads already have the target company name.")

# Verify
print("\n--- VERIFICATION OF ALL LEADS IN ZOHO CRM ---")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    for l in data.get("data", []):
        print(f"ID: {l['id']} | Name: {l.get('Full_Name')} | Company: {l.get('Company')} | Email: {l.get('Email')}")

