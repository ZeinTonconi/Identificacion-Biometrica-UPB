const express = require('express');
const prisma = require('./db');

const app = express();
const port = 4000;

app.use(express.json());

// Endpoint para user_types
app.get('/user_types', async (req, res) => {
  try {
    const userTypes = await prisma.user_types.findMany();
    res.json(userTypes);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error querying user_types' });
  }
});

// Endpoint para users
app.get('/users', async (req, res) => {
  try {
    const users = await prisma.users.findMany({
      include: { user_type: true },
    });
    res.json(users);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error querying users' });
  }
});

// Endpoint para faces
app.get('/faces', async (req, res) => {
  try {
    const faces = await prisma.faces.findMany({
      include: { user: true },
    });
    res.json(faces);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error querying faces' });
  }
});

// Endpoint para devices
app.get('/devices', async (req, res) => {
  try {
    const devices = await prisma.devices.findMany();
    res.json(devices);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error querying devices' });
  }
});

// Endpoint para access_logs
app.get('/access_logs', async (req, res) => {
  try {
    const logs = await prisma.access_logs.findMany({
      include: { user: true, device: true },
    });
    res.json(logs);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error querying access_logs' });
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});