import os, requests, time
from urllib.parse import urlencode

class TokenManager:
    def __init__(self):
        self.expiry_time = None
        self.access_token = None

    def get_authorization_url(self):

        tenant_id = os.getenv("MS_TENANT_ID")
        client_id = os.getenv("MS_CLIENT_ID")  # safer to use env vars
        client_secret = os.getenv("MS_CLIENT_SECRET")
        redirect_uri = "http://localhost:8080"
        # The scopes you’re requesting (can be multiple, space-separated)
        scopes = [
            "User.Read",
            "Notes.ReadWrite"
        ]

        # Build the authorization URL manually
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "response_mode": "query",
            "scope": " ".join(scopes),
            "state": "12345"  # optional random string for CSRF protection
        }

        auth_url = (
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize?"
            + urlencode(params)
        )

        print("Visit this URL to authorize your app:\n")
        print(auth_url)
        print("\nAfter authorizing, you'll be redirected to your redirect_uri.")
        print("Copy the full URL from the address bar and paste it into your app.")
        code=input("enter code:\t")
        return code


    def get_token(self):
        tenant_id = os.getenv("MS_TENANT_ID")
        client_id = os.getenv("MS_CLIENT_ID")  # safer to use env vars
        client_secret = os.getenv("MS_CLIENT_SECRET")
        redirect_uri = "http://localhost:8080"        
        if self.access_token and time.time() > self.expiry_time:
            return self.access_token
        auth_code = self.get_authorization_url()  # from your authorization step

        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

        data = {
            "client_id": client_id,
            "scope": "https://graph.microsoft.com/.default",
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "client_secret": client_secret,
        }

        response = requests.post(token_url, data=data)
        response.raise_for_status()

        self.access_token = response.json()['access_token']
        self.expiry_time = response.json()['expires_in'] = time.time()
        return self.access_token