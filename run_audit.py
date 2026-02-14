import requests

url = "http://localhost:8000/api/audit"
golden_source = "Enterprise_Vendor_Governance_Policy.pdf"
audit_file_path = r"c:\Automations\BABOK RAG\data\raw\temp_Vendor_Renewal_Contract_NonCompliant.pdf"

with open(audit_file_path, "rb") as f:
    files = {"audit_file": f}
    data = {"golden_source": golden_source}
    
    print(f"🚀 Running Audit comparison...")
    print(f"Golden Source: {golden_source}")
    print(f"Target Draft: {audit_file_path}\n")
    
    try:
        response = requests.post(url, data=data, files=files, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        print("✅ AUDIT COMPLETE\n")
        print("AI RESPONSE:")
        print("-" * 30)
        print(result["result"]["answer"])
        print("-" * 30)
    except Exception as e:
        print(f"❌ Error during audit: {e}")
