import json

def main():
    try:
        with open("mbc.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        behaviors = {}
        # STIX 2.1 JSON structure: "objects" is a list of STIX objects
        for obj in data.get("objects", []):
            if obj.get("type") in ["malware-behavior", "malware-objective", "malware-method"]:
                name = obj.get("name", "")
                raw_desc = obj.get("description", obj.get("obj_defn", ""))
                
                # obj_defnが辞書型の場合、中のdescriptionキーからテキストを取り出す
                if isinstance(raw_desc, dict):
                    desc = raw_desc.get("description", "")
                else:
                    desc = raw_desc
                
                # Sentence-BERT can handle up to 256/512 tokens. Keeping full is fine.
                if name and desc and isinstance(desc, str):
                    behaviors[name] = desc
        
        print(f"Extracted {len(behaviors)} MBC behaviors.")
        
        with open("src/mbc_full_categories.json", "w", encoding="utf-8") as f:
            json.dump(behaviors, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
