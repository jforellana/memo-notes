# def get_token():
#     from requests_oauthlib import OAuth2Session
#     from dotenv import load_dotenv
#     import os
#     # load_dotenv()

#     os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
#     # client_id = os.environ['MS_CLIENT_ID']
#     client_id = os.getenv("MS_CLIENT_ID")
#     # client_secret = os.environ['MS_CLIENT_SECRET']
#     client_secret = os.getenv("MS_CLIENT_SECRET")
#     redirect_uri = "http://localhost:8080"

#     scope = ["https://graph.microsoft.com/.default"]
#     oauth = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=scope)
#     auth_url, state = oauth.authorization_url(
#         "https://login.microsoftonline.com/c6fc6e9b-51fb-48a8-b779-9ee564b40413/oauth2/authorize")
    
#     print(f"Please go to {auth_url} and authorize access")
#     auth_response = input("Enter callback URL:\t")

#     token = oauth.fetch_token(
#         "https://login.microsoftonline.com/c6fc6e9b-51fb-48a8-b779-9ee564b40413/oauth2/v2.0/token",
#         authorization_response=auth_response,
#         client_id=client_id,
#         client_secret=client_secret
#     )

#     return token

import os, requests
from urllib.parse import urlencode

def get_authorization_url():
    tenant_id = "c6fc6e9b-51fb-48a8-b779-9ee564b40413"
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



import os
import requests

def get_token():
    tenant_id = "c6fc6e9b-51fb-48a8-b779-9ee564b40413"
    client_id = os.getenv("MS_CLIENT_ID")  # safer to use env vars
    client_secret = os.getenv("MS_CLIENT_SECRET")
    redirect_uri = "http://localhost:8080"
    auth_code = get_authorization_url()  # from your authorization step

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
    return response.json()
