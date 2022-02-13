import React from 'react';
import Button from '@mui/material/Button';
import CssBaseline from '@mui/material/CssBaseline';
import TextField from '@mui/material/TextField';
import Link from '@mui/material/Link';
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';
import './App.css';
import { BASE_URL, environment } from './routeConstants';

function App() {
  const handleSubmit = (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    console.log(data.get('inputstring'))
    fetch(`${BASE_URL[environment]}/api/classify`, {
      crossDomain: true,
      method: 'POST',
      body: JSON.stringify({ inputstring: data.get('inputstring')}),
      headers: {
      'Content-type': 'application/json'
      }
  })
      .then(response => {
          console.log(response)
      });
  };

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
            <Grid container>
              <Grid item xs>
                <Link href="#" variant="body2">
                  GitHub
                </Link>
              </Grid>
              <Grid item>
                <Link href="#" variant="body2">
                  {"Documentation"}
                </Link>
              </Grid>
            </Grid>
          </Box>
        </Box>
      </Container>
  );
}

export default App;
