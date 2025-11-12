from flask import Flask, request
import subprocess
import html  # built-in module for escaping HTML

app = Flask(__name__)

# HTML template with optional previous output and image
def html_page(previous_output="", image_url=""):
    image_html = f'<h3>Image Preview:</h3><img src="{html.escape(image_url)}" width="300"><br><br>' if image_url else ""
    output_html = f'<h3>Previous Prediction:</h3><pre style="background:#111;color:#0f0;padding:1em;">{html.escape(previous_output)}</pre>' if previous_output else ""
    return f"""
    <h2>Garbage Classification</h2>
    <p>Hello and welcome to my project! In it I'll be analyzing and training a model on a dataset about classification of garbage.</p>
    <p>This dataset is from Kaggle, https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download.</p>
    <p>In it there are many images of garbage and we'll train the model to be able to distribute which of them is: glass, metal, paper, plastic, or other trash.</p>
    <p>I've analyzed the dataset and I've come to the conslusion that the data is pretty well distributed between the different classes of garbage and therefore the dataset is fair to work with.</p>
    <p>In this project, you can enter the url of an image from the internet and the model will predict its class. The accuracy of the model is around 30% (checked in the other files that can be found in the GitHub repository).</p>
    <p>After hitting the button "Predict" you'll see the image and the made prediction in the console.</p>
    <p>If you want to get a "prediction" for your personal image, please upload it to a website like https://postimages.org/ and then paste the url to this application.</p>
    <form action="/run">
        Image URL: <input type="text" name="url" size="60" value="{html.escape(image_url)}">
        <input type="submit" value="Predict">
    </form>
    {image_html}
    {output_html}
    """

@app.route('/')
def index():
    return html_page()

@app.route('/run')
def run_prediction():
    url = request.args.get('url', '').strip()
    if not url:
        return "<p>Error: Please provide an image URL.</p>"

    # Start subprocess with the URL
    process = subprocess.Popen(
        ["python3", "main_train_only.py", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line)
    process.stdout.close()
    process.wait()

    output_text = "".join(output_lines)

    # Return HTML with form, image, and prediction
    return html_page(previous_output=output_text, image_url=url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

    
