from flask import Flask, request, json
from flask_cors import CORS, cross_origin
import api_model
import os
import numpy as np

model_path = 'model_cl'
clarifai_api_key = '***'
google_vision_credentials = '***'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_vision_credentials
CLF = api_model.clarifai(clarifai_api_key)
TEX = api_model.texify()
CLA = api_model.classify(model_path)

#################

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
@cross_origin()
def query_example():
    url = request.args.get('url')

    caption = TEX.caption_detect(url=url)
    # caption = 'My wife should be fully covered by a comprehensive insurance policy.'
    print(caption)

    X = CLF.embed_all(url=url, caption=caption)

    X1 = np.asarray(X['txt'])
    X2 = np.asarray(X['txt_mod'])
    X3 = np.asarray(X['img_txt'])
    X4 = np.asarray(X['img'])
    X5 = np.asarray(X['img_mod'])

    X = np.concatenate((X1, X2, X3, X4, X5), axis=0)
    X = X.reshape(1, X.shape[0])

    pred = CLA.predict(X)

    print(pred)

    return {
        'statusCode': 200,
        'body': json.dumps(str(pred))
    }


if __name__ == "__main__":
    app.run(debug=True)
