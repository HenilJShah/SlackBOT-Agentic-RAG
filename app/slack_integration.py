import os
import logging
from flask import Flask, request
from slack_bolt import App
from slack_bolt.oauth.oauth_settings import OAuthSettings
from slack_sdk.oauth.installation_store import FileInstallationStore
from slack_sdk.oauth.state_store import FileOAuthStateStore
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk import WebClient
from waitress import serve
import requests
import json

# Define constants
INSTALLATIONS_DIR = "./data/installations/"
STATES_DIR = "./data/states/"
CLIENT_ID = os.environ["CLIENT_ID"]
CLIENT_SECRET = os.environ["CLIENT_SECRET"]
SIGNING_SECRET = os.environ["SIGNING_SECRET"]
SLACK_TOKEN = os.environ["SLACK_TOKEN"]
FIXED_API_ENDPOINT = os.environ["API_ENDPOINT"]


url = "https://afda4bfb-11e7-4f56-b4b9-72ebdad8b636-00-stf60978uc6q.pike.replit.dev/ask"
headers = {"Content-Type": "application/json"}


# Set up directories
def setup_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    os.chmod(dir_path, 0o755)


setup_directory(INSTALLATIONS_DIR)
setup_directory(STATES_DIR)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# OAuth settings and app initialization
oauth_settings = OAuthSettings(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scopes=["chat:write", "im:write", "im:history", "chat:write.public", "commands"],
    installation_store=FileInstallationStore(base_dir=INSTALLATIONS_DIR),
    state_store=FileOAuthStateStore(expiration_seconds=600, base_dir=STATES_DIR),
    redirect_uri="https://cfd4b4b1-2826-4f04-a9be-0d513123fea1-00-lvme8yan71kf.pike.replit.dev/slack/oauth_redirect",
)

app = App(signing_secret=SIGNING_SECRET, oauth_settings=oauth_settings)

# WebClient for sending messages
client = WebClient(token=SLACK_TOKEN)


# Event listener for DMs (Direct Messages) to the bot
@app.event("message")
def handle_direct_message(body, event, say):
    if event.get("channel_type") == "im":
        message_text = event["text"]

        data = {"question": message_text}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response.status_code)  # Print the status code
        print(response.json())  # Print the response JSON (if any)
        # OPTIONAL: add 'thinking' message
        # say("Thinking...")
        # response = requests.get(FIXED_API_ENDPOINT + message_text)
        # response_text = response.json(
        # )['response'] if response.status_code == 200 else f"Request failed with status code {response.status_code}"

        say(str(response.json()["answer"]))


# Flask app and routes
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)


@flask_app.route("/slack/install", methods=["GET"])
def install():
    return handler.handle(request)


@flask_app.route("/slack/oauth_redirect", methods=["GET"])
def oauth_redirect():
    return handler.handle(request)


@flask_app.route("/")
def hello_world():
    return "Hello from the Slack bot instance! Now trying OAuth"


# Run the app on Waitress server
if __name__ == "__main__":
    serve(flask_app, host="0.0.0.0", port=81)
