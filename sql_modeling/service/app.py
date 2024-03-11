from flask import Flask, request, render_template_string, send_file
import subprocess

app = Flask(__name__)

params = [
{
	'name': "pop",
	'default': "int(1e7)+1",
        'description': "Total human population (not agents)"
},
{
        'name': 'num_nodes',
	'default': "60",
        'description': "Number of nodes/populations"
},
{
        'name': 'eula_age',
	'default': "5",
        'description': "Age at which we assume initial population is Epidemiologically Uninteresting (Immune)"
},
{
        'name': 'simulation duration in years',
	'default': "1",
        'description': "Like it sounds"
},
{
        'name': 'base_infectivity',
	'default': "1.5e7",
        'description': "Proxy for R0"
},
{
        'name': 'cbr',
	'default': "15",
        'description': "Crude Birth Rate (same for all nodes)"
},
{
        'name': 'campaign_day',
	'default': "60",
        'description': "Day at which one-time demo campaign occurs"
},
{
        'name': 'campaign_coverage',
	'default': "0.75",
        'description': "Coverage to use for demo campaign"
},
{
        'name': 'campaign_node',
	'default': "15",
        'description': "Node to target with demo campaign"
},
{
        'name': 'migration_interval',
	'default': "7",
        'description': "Timesteps to wait being doing demo migration"
},
{
        'name': 'mortality_interval',
	'default': "7",
        'description': "Timesteps between applying non-disease mortality."
},
{
        'name': 'fertility_interval',
	'default': "7",
        'description': "Timesteps between adding new babies."
},
{
        'name': 'ria_interval',
	'default': "7",
        'description': "Timesteps between applying routine immunization of 9-month-olds."
},
]


# Define the API documentation
API_DOC = """
<h1>Web Service API</h1>
<p>This web service allows you to submit parameters and run the application.</p>
<p>To submit parameters, make a POST request to /submit with the following parameters:</p>
<ul>
  <li>param1: description of parameter 1</li>
  <li>param2: description of parameter 2</li>
  <!-- Add more parameters as needed -->
</ul>
"""

# Define the HTML template for the form
FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Web Service Form</title>
</head>
<body>
  <h1>Submit Parameters</h1>
  <form id="myForm" method="POST" action="/submit">
    <!-- Dynamically generate form fields -->
    <div id="formFields"></div>
    <button type="submit">Submit</button>
  </form>

  <!-- Div to display the returned image -->
  <div id="imageContainer"></div>

  <!-- JavaScript to handle form submission and image display -->
  <script>
    // Define the params list of dictionaries
    var params = [
{
	'name': "pop",
	'default': "int(1e7)+1",
        'description': "Total human population (not agents)"
},
{
        'name': 'num_nodes',
	'default': "60",
        'description': "Number of nodes/populations"
},
{
        'name': 'eula_age',
	'default': "5",
        'description': "Age at which we assume initial population is Epidemiologically Uninteresting (Immune)"
},
{
        'name': 'simulation duration in years',
	'default': "1",
        'description': "Like it sounds"
},
{
        'name': 'base_infectivity',
	'default': "1.5e7",
        'description': "Proxy for R0"
},
{
        'name': 'cbr',
	'default': "15",
        'description': "Crude Birth Rate (same for all nodes)"
},
{
        'name': 'campaign_day',
	'default': "60",
        'description': "Day at which one-time demo campaign occurs"
},
{
        'name': 'campaign_coverage',
	'default': "0.75",
        'description': "Coverage to use for demo campaign"
},
{
        'name': 'campaign_node',
	'default': "15",
        'description': "Node to target with demo campaign"
},
{
        'name': 'migration_interval',
	'default': "7",
        'description': "Timesteps to wait being doing demo migration"
},
{
        'name': 'mortality_interval',
	'default': "7",
        'description': "Timesteps between applying non-disease mortality."
},
{
        'name': 'fertility_interval',
	'default': "7",
        'description': "Timesteps between adding new babies."
},
{
        'name': 'ria_interval',
	'default': "7",
        'description': "Timesteps between applying routine immunization of 9-month-olds."
}
    ];

    // Function to create text entry fields based on params list
    function createFormFields() {
      var formFieldsHtml = '';
      params.forEach(function(param) {
        formFieldsHtml += '<label for="' + param.name + '">' + param.name + ':</label>';
        formFieldsHtml += '<input type="text" name="' + param.name + '" id="' + param.name + '" value="' + param.default + '"><br>';
        formFieldsHtml += '<small>' + param.description + '</small><br><br>';
      });
      document.getElementById('formFields').innerHTML = formFieldsHtml;
    }

    // Call createFormFields function when the page loads
    window.onload = createFormFields;

    // Function to handle form submission
    document.getElementById('myForm').addEventListener('submit', function(event) {
      event.preventDefault(); // Prevent default form submission

      // Serialize form data
      var formData = new FormData(this);

      // Send form data using AJAX
      var xhr = new XMLHttpRequest();
      xhr.open('POST', '/submit', true);
      xhr.onload = function() {
        if (xhr.status === 200) {
          // If request is successful, display the returned image
          var imageUrl = 'prevs.png'; // Replace with actual URL of returned image
          document.getElementById('imageContainer').innerHTML = '<img src="' + imageUrl + '" alt="prevs">';
        }
      };
      xhr.send(formData);
    });
  </script>
</body>
</html>
""" 

def run_sim():
    import sir_numpy_c as model
    import settings
    import report
    from tlc import run_simulation
    ctx = model.initialize_database()
    ctx = model.eula_init( ctx, settings.eula_age )

    csv_writer = report.init()

    # Run the simulation for 1000 timesteps
    try:
        subprocess.run(['/usr/local/bin/python3', 'tlc.py'], check=True)
    except subprocess.CalledProcessError as e:
        # Handle subprocess error (e.g., log the error, restart the subprocess)
        print(f"Subprocess error: {e}")
    #from functools import partial
    #runsim = partial( run_simulation, ctx=ctx, csvwriter=csv_writer, num_timesteps=settings.duration )
    #runsim()
    import plot_spatial
    plot_spatial.load_and_plot( csv_file="simulation_output.csv" )

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # Process the submitted parameters
        data = request.form
        # Run your application logic here
        base_infectivity = float(data['base_infectivity'])
        cbr = int(data['cbr'])
        #return 'Data received: {}'.format(data)
        run_sim()
        #return f'Sim ran'
        return {'url': '/prevs.png'}
    else:
        # Return the API documentation
        return API_DOC

@app.route('/prevs.png')
def download_prevs():
    return send_file("./prevs.png", as_attachment=True)

@app.route('/')
def index():
    # Generate the HTML form with parameters
    # This could be dynamically generated based on the parameters your application expects
    form_html = render_template_string(FORM_TEMPLATE, params=params)
    return form_html

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
"""
    {% for param, desc in params.items() %}
      <label for="{{ param }}">{{ param }}:</label>
      <input type="text" name="{{ param }}" id="{{ param }}"><br>
      <small>{{ desc }}</small><br><br>
    {% endfor %}
"""
