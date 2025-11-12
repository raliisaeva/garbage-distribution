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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    