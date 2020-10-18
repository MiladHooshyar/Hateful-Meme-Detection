from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from google.cloud import vision
import os
import tensorflow as tf

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'theta-totem-242819-bb71dfad2b1d.json'


class clarifai:

    def __init__(self, app_id):
        self.app_id = app_id
        self.channel = ClarifaiChannel.get_json_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)

    ########################
    def img_embed(self, url):
        request = service_pb2.PostModelOutputsRequest(
            model_id='bbb5f41425b8468d9b7a554ff10f8581',
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=url)))
            ])
        metadata = (('authorization', self.app_id),)
        response = self.stub.PostModelOutputs(request, metadata=metadata)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Request failed, status code: " + str(response.status.code))

        return response.outputs[0].data.embeddings[0].vector

    #######################
    def img_mod_embed(self, url):
        request = service_pb2.PostModelOutputsRequest(
            model_id='d16f390eb32cad478c7ae150069bd2c6',
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=url)))
            ])
        metadata = (('authorization', self.app_id),)
        response = self.stub.PostModelOutputs(request, metadata=metadata)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Request failed, status code: " + str(response.status.code))
        return [x.value for x in response.outputs[0].data.concepts]

    ######################################
    def img_txt_embed(self, url, caption):
        request = service_pb2.PostModelOutputsRequest(
            model_id='aaa03c23b3724a16a56b629203edc62c',
            inputs=[
                resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=url)))
            ])
        metadata = (('authorization', self.app_id),)
        response = self.stub.PostModelOutputs(request, metadata=metadata)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Request failed, status code: " + str(response.status.code))
        img_cons = ' '.join([x.name for x in response.outputs[0].data.concepts])
        return self.txt_embed(caption + '. ' + ' '.join(img_cons.split(' ')[:10]))

    #############################
    def txt_embed(self, caption):
        response = self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                model_id="568d48e82924a00d0f98a6d34fa426cf",
                inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=caption)))]
            ),
            metadata=(('authorization', self.app_id),)
        )
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Request failed, status code: " + str(response.status.code))
        return response.outputs[0].data.embeddings[0].vector

    #################################

    def txt_mod_embed(self, caption):
        response = self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                model_id="39f2950a32173f61b3eb40ede0d254e1",
                inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=caption)))]
            ),
            metadata=(('authorization', self.app_id),)
        )
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception("Request failed, status code: " + str(response.status.code))
        return response.outputs[0].data.embeddings[0].vector

    def embed_all(self, url, caption):
        X_img = self.img_embed(url)
        X_img_mod = self.img_mod_embed(url)
        X_txt = self.txt_embed(caption)
        X_txt_mod = self.txt_mod_embed(caption)
        X_img_txt = self.img_txt_embed(url, caption)
        return {'img': X_img, 'img_mod': X_img_mod,
                'txt': X_txt, 'txt_mod': X_txt_mod,
                'img_txt': X_img_txt}


class texify:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
        self.image = vision.Image()

    def caption_detect(self, url):
        self.image.source.image_uri = url
        response = self.client.text_detection(image=self.image)
        texts = response.text_annotations
        if response.error.message:
            raise Exception(
                'Error'.format(
                    response.error.message))

        if len(texts) > 0:
            return texts[0].description
        else:
            return None


class classify:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, X):
        return self.model.predict(X)[0][0]
