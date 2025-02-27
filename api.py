from flask import Flask, request, jsonify, send_file
import os
from PIL import Image
import io
import generator

app = Flask(__name__)

@app.route('/run-genetic-algorithm', methods=['POST'])
def run_genetic_algorithm():
    try:
        # Get parameters from the request
        data = request.form

        pop_size = int(data.get('pop_size', 100))  # Default to 100 if not provided
        num_generations = int(data.get('num_generations', 2001))  # Default to 2001
        mutation_rate = float(data.get('mutation_rate', 0.1))  # Default to 0.1

        # Get the uploaded image file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        target_image = Image.open(image_file)

        # Run the genetic algorithm with the provided parameters
        reconstructed_image = generator.genetic_algorithm(pop_size, num_generations, mutation_rate, target_image)

        # Save the result to a bytes buffer
        output_buffer = io.BytesIO()
        reconstructed_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        # Return the generated image as a response
        return send_file(output_buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)