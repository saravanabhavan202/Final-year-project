import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TMP_FOLDER'] = 'static/tmp'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MODEL_PATH'] = 'alzheimer_model.keras'
app.config['CLASS_NAMES'] = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
app.config['T_THRESHOLDS'] = [0.1, 0.3, 0.5, 0.7, 0.9]
app.config['BRAIN_ROIS'] = {
    'Hippocampus': [(80, 140, 40, 20), (140, 140, 40, 20)],
    'Temporal Lobe': [(40, 100, 60, 80), (140, 100, 60, 80)],
    'Frontal Lobe': [(80, 20, 60, 60)]
}
app.config['IMG_SIZE'] = (224, 224)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TMP_FOLDER'], exist_ok=True)

model = load_model(app.config['MODEL_PATH'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    result = process_image(upload_path)
    return render_template('result.htmll',
                           original_image=filename,
                           predicted_class=result['predicted_class'],
                           confidence_scores=zip(app.config['CLASS_NAMES'], result['confidence_scores']),
                           plot_filename=result['plot_filename'],
                           roi_results=result['roi_results'],
                           thresholds=app.config['T_THRESHOLDS'])


def preprocess_image(image):
    image = tf.image.resize(image, app.config['IMG_SIZE'])
    return tf.image.convert_image_dtype(image, tf.float32)


def compute_feature_importance(model, image_batch, class_idx):
    with tf.GradientTape() as tape:
        tape.watch(image_batch)
        preds = model(image_batch)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, image_batch)
    importance = tf.abs(grads)
    importance = tf.reduce_max(importance, axis=-1)[0]

    # Use TensorFlow functions instead of .min() and .max()
    min_val = tf.reduce_min(importance)
    max_val = tf.reduce_max(importance)

    return (importance - min_val) / (max_val - min_val + 1e-8)

def analyze_rois(mask):
    results = {}
    for roi, boxes in app.config['BRAIN_ROIS'].items():
        total = selected = 0
        for (x, y, w, h) in boxes:
            region = mask[y:y + h, x:x + w]
            total += w * h
            selected += np.sum(region)
        results[roi] = selected / total if total > 0 else 0
    return results


def generate_plot(image_display, importance_map, prediction, confidence, class_idx):
    fig, axes = plt.subplots(1, len(app.config['T_THRESHOLDS']) + 1, figsize=(20, 5))
    axes[0].imshow(image_display)
    axes[0].set_title(f"{prediction}\nConfidence: {confidence[class_idx]:.1f}%")
    axes[0].axis('off')

    for i, t in enumerate(app.config['T_THRESHOLDS']):
        mask = importance_map > t
        axes[i + 1].imshow(image_display, alpha=0.5)
        axes[i + 1].imshow(np.where(mask, importance_map, 0), cmap='hot', alpha=0.5)

        roi_significance = analyze_rois(mask)
        for roi, sig in roi_significance.items():
            if sig > 0.3:
                for (x, y, w, h) in app.config['BRAIN_ROIS'][roi]:
                    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='cyan', facecolor='none')
                    axes[i + 1].add_patch(rect)
                    axes[i + 1].text(x, y - 2, roi, color='cyan', fontsize=6, weight='bold')

        axes[i + 1].set_title(f"Threshold: {t}\nSelected: {mask.mean():.2%}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(app.config['TMP_FOLDER'], f'plot_{uuid.uuid4()}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path


def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_display = img.resize(app.config['IMG_SIZE'])
    img_array = img_to_array(img)
    img_tensor = preprocess_image(tf.convert_to_tensor(img_array))

    preds = model.predict(tf.expand_dims(img_tensor, 0))[0]
    class_idx = np.argmax(preds)
    confidence = (preds * 100).tolist()

    importance_map = compute_feature_importance(model, tf.expand_dims(img_tensor, 0), class_idx)
    plot_path = generate_plot(img_display, importance_map.numpy(),
                              app.config['CLASS_NAMES'][class_idx], confidence, class_idx)

    roi_results = {}
    for t in app.config['T_THRESHOLDS']:
        mask = importance_map > t
        roi_results[t] = {
            'selection_rate': mask.numpy().mean(),
            'roi_significance': analyze_rois(mask.numpy())
        }

    return {
        'predicted_class': app.config['CLASS_NAMES'][class_idx],
        'confidence_scores': confidence,
        'plot_filename': os.path.basename(plot_path),
        'roi_results': roi_results
    }


if __name__ == '__main__':
    app.run(debug=True)