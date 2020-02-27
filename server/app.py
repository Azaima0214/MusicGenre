from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import ssl

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
key_path = "your_key_path"

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/predict" , methods=['POST', 'OPTIONS'])
def predict():

	if request.method == "OPTIONS": # CORS Preflight
		return _build_cors_prelight_response()
	elif request.method == "POST":
		file = request.files['file']
		fileName = file.filename
		file.save("your_directry" + fileName)

		import predict_music
		pr = predict_music.predict_music()
		pr.load_file(fileName)
		result= pr.predict_genre()
		response =  jsonify({'genre' : result})
		return _corsify_actual_response(response)

	return "Error"

def _build_cors_prelight_response():
	response = make_response()
	response.headers.add("Access-Control-Allow-Origin", "*")
	response.headers.add('Access-Control-Allow-Headers', "*")
	response.headers.add('Access-Control-Allow-Methods', "*")
	return response

def _corsify_actual_response(response):
	response.headers.add("Access-Control-Allow-Origin", "*")
	return response

if __name__ == '__main__':
	ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
	ssl_context.load_cert_chain(key_path + 'your_key2.pem', key_path + 'your_key2.pem')
	app.run(ssl_context=ssl_context)
