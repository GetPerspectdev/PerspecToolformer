import os
from flask import Flask, jsonify, request
from calculate import run_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET"])
def say_hello():
    return jsonify({"msg": "Hello from the Professionalism API!"})

@app.route("/score/professionalism", methods=["POST"])
def get_professionalism():
    body = request.json
    messages = body['messages']
    return run_pipeline(message_history=messages)


if __name__ == "__main__":
    debug = os.environ.get("DEBUG", False)
    app.run(host="0.0.0.0", port=5050, debug=debug)