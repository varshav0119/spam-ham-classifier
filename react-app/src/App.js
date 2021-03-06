import React, { useState } from 'react';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import './App.css';
import { BASE_URL, environment } from './routeConstants';

function App() {
  var [result, setResult] = useState({});
  const handleSubmit = (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    console.log(data.get('inputstring'))
    fetch(`${BASE_URL[environment]}/api/classify/`, {
      crossDomain: true,
      method: 'POST',
      body: JSON.stringify({ inputstring: data.get('inputstring')}),
      headers: {
      'Content-type': 'application/json'
      }
    })
    .then(response => response.json())
    .then(response => {
      console.log(response)
      setResult(result => {
        return({
            classification: response.classification,
            inputstring: data.get('inputstring'),
            spamScore: response.spam_score
          })
      })
    })
  }

  return (
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Typography component="h1" variant="h5">
            Spam Classification
          </Typography>
          <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="inputstring"
              label="Input String"
              name="inputstring"
              autoFocus
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
            >
              Is this Spam?
            </Button>
            {Object.keys(result).length?
              (result.classification === "NOT SPAM"?
                <Alert severity="success">
                <AlertTitle><strong>{result.classification}</strong></AlertTitle>
                  <strong>Spam Score: {result.spamScore.toFixed(2)}</strong>
                  <div>
                    {result.inputstring}
                  </div>
                </Alert>
              :
                <Alert severity="error">
                <AlertTitle><strong>{result.classification}</strong></AlertTitle>
                  <strong>Spam Score: {result.spamScore.toFixed(2)}</strong>
                  <div>
                    {result.inputstring}
                  </div>
                </Alert>)
            :
                <></>
            }
            <Grid container>
              <Grid item xs>
                <Link href="https://github.com/varshav0119/spam-ham-classifier" variant="body2">
                  GitHub
                </Link>
              </Grid>
              <Grid item>
                <Link href="https://varshav0119.notion.site/Spam-Classification-180fba0bc8ce49a78747e6eb9a66b09a" variant="body2">
                  Report
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
      </Container>
  );
}

export default App;
