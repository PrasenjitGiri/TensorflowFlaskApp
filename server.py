
import tensorflow as tf
import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'


def load_graph(graph_filename):
    with tf.gfile.FastGFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    return graph


@app.route('/classify', methods=['GET'])
@cross_origin()
def classify_image():
    result = {}
    filename = request.args.get('file')

    if filename:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        predictions = persistent_session.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        low_confidence = 0
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score < 0.90:
                low_confidence += 1
            result[human_string] = str(score)

        if low_confidence >= 2:
            result['error'] = 'Unable to classify document type (Passport/Driving License)'

    return jsonify(result)


if __name__ == '__main__':
    tf_graph = load_graph('output_graph.pb')
    softmax_tensor = tf_graph.get_tensor_by_name('final_result:0')
    label_lines = [line.strip() for line in tf.gfile.GFile("output_labels.txt")]
    persistent_session = tf.Session(graph=tf_graph)
    app.run(port=9900, debug=True)
