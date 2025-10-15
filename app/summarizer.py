def summarize(originalTranscript, api_key, bulk):
    from openai import OpenAI
    import json

    client = OpenAI(api_key=api_key)

    def summarize_and_organize(title, summary, section):
        from token_generator import get_token
        import pytz
        from datetime import datetime
        import requests

        base_url = "https://graph.microsoft.com/v1.0/users/641adf28-842a-438f-8340-61c0f0b9c9d6/onenote/sections/"

        token = get_token()

        header = {"Content-Type": "application/xhtml+xml",
                  "Authorization": f"Bearer {token['access_token']}"}

        sections = {
            "other": "1-2aa817a9-4887-4597-865f-5b4060ec7d9c",
            "healthcare leadership lessons": "1-636bd91a-f9c9-4a76-b902-67160b2883c5",
            "religion": "1-da3eaac8-e6b9-4740-9674-bb9345b5de86",
            "capstone": "1-9c2a0c24-f044-4a41-a938-ac638a31f3e3",
            "dynamic of US healthcare": "1-e21c8d7c-925a-4ac4-b69b-a4a2f7970ee8",
            "healthcare policy": "1-59147fec-c548-401a-9ae5-22ee4f800b74",
            "strategy": "1-92638758-e6d1-45e2-ac30-b22009b9a4cb",
            "natural language processing": "1-8a58de88-5d4f-4998-b9aa-bb5fe933d93e"
        }

        # sectionNames = [k for k,v in sections.items()]
        # if bulk:
        #     print(f"Title: {title}")
        # section=input(f"What section should this note go into? {sectionNames}:\t")
        section = sections.get(section)

        tz = datetime.now(pytz.timezone("America/Denver"))


        html_body="""
<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<meta name="created" content="{timezone}" />
</head>
<body>
{summary}
</body>
</html>
"""
        
        new_note = requests.post(
            f"{base_url}{section}/pages",
            headers=header,
            data=html_body.format(title=title, summary=summary, timezone=tz))
        
        return new_note
        

    prompt = """Take the transcript provided below and create a markdown summary
        with sections for highlights, key notes, reminders, alerts and more, depending on the
        content. Send the summary to OneNote.
        Take the transcript and create:
        1. The title.
        2. The content. Do not include the title in the content.
        3. Select the section/category that best fits the content.


        Transcript: {transcript} 
        """
    
    input_list = [{"role": "user", "content": prompt.format(transcript=originalTranscript)}]
    
    tools = [
        {
            "type": "function",
            "name": "summarize_and_organize",
            "description": "Summarize the transcript and send it to OneNote.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary for the transcript"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the summary"
                    },
                    "section": {
                        "type": "string",
                        "description": "Section or category for the class",
                        "enum": ["healthcare leadership lessons",
                                 "religion",
                                 "capstone",
                                 "dynamics of US healthcare",
                                 "healthcare policy",
                                 "strategy",
                                 "natural language processing",
                                 "other"]
                    }
                },
                "required": ["summary", "title", "section"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]
        
    response = client.responses.create(
        model = "gpt-5",
        input=input_list,
        tools=tools
    )


    for tool_call in response.output:
        if tool_call.type != "function_call":
            continue

        name = tool_call.name
        args = json.loads(tool_call.arguments)
        result = summarize_and_organize(**args)

        input_list.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        })
