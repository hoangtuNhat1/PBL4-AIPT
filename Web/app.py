from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
from process import deadlift, squat, bicep_curl, lunge, plank
from static_remover import clear_folder
from process import squat
from down_res import reduce_video_quality
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static\\videos\\'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def convert_image_name(image_name):
    parts = image_name.split('_')
    formatted_name = " ".join(parts).replace('At', 'at').replace('-', ':').replace('.jpg', '')
    return formatted_name
def get_image_list():
    error_images = os.listdir('static/images')  # Đổi đường dẫn tùy theo cách bạn đặt thư mục
    return error_images
@app.route('/')
def index():
    image_names = get_image_list()
    formatted_image_names = [convert_image_name(name) for name in image_names]
    out_path = session.get('out_path', None)
    # out_path =     'uploads/upload.mp4'
    # context = {
    #     'out_path': out_path
    # }
    error_count = session.get('error_count', None)
    total_error_count = session.get('total_error_count', None)
    return render_template('index.html',out_path = out_path, error_count=error_count, total_errors = total_error_count, image_names = image_names, formatted_image_names = formatted_image_names )
    # return render_template('index.html',out_path=out_path)

@app.route('/upload', methods=['POST'])
def upload():
    clear_folder("uploads")
    clear_folder("static/images")
    clear_folder("static/videos")
    
    uploaded_file = request.files['file']
    exercise = request.form.get('exercise')
    
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "upload.mp4")
        uploaded_file.save(file_path)
    
    if exercise == "deadlift":
        session['out_path'], session['error_count'], session['total_error_count'] =  deadlift(file_path)
    elif exercise == "squat":
        session['out_path'], session['error_count'], session['total_error_count'] =  squat(file_path)
    elif exercise == "bicep_curl":
        session['out_path'], session['error_count'], session['total_error_count'] =  bicep_curl(file_path)
    elif exercise == "lunge":
        session['out_path'], session['error_count'], session['total_error_count'] =  lunge(file_path)
    elif exercise == "plank":
        session['out_path'], session['error_count'], session['total_error_count'] =  plank(file_path)
    reduce_video_quality(session['out_path'], 'static\\videos\\output_compress.mp4')
    session['out_path'] = 'static\\videos\\output_compress.mp4'
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
