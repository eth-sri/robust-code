const express = require('express');
const bodyParser = require('body-parser');
const app = express();

let ArgumentParser = require('argparse').ArgumentParser;
let parser = new ArgumentParser({
  addHelp:true,
  description: 'TypeScript Parser'
});
parser.addArgument(
  [ '--port' ],
  {
    help: 'Port on which to start server',
    defaultValue: 3000
  }
);
let args = parser.parseArgs();

const port = args.port;
const parse = require('./server_parse');

app.use(bodyParser.json({limit: '50mb'}));

app.post('/api/v1/parse', (request, response) => {
    console.log(port + ", " + request.body.filename);
    let remove_types = 'remove_types' in request.body ? request.body.remove_types : false;
    let dependencies = 'deps' in request.body ? request.body.deps : [];
    let data = parse.parse_file(request.body.filename, remove_types, dependencies);
    response.send(data);
});

app.listen(port, (err) => {
    if (err) {
        return console.log('something bad happened', err);
    }

    console.log(`server is listening on ${port}`);
});