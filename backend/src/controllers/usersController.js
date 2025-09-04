// controllers/usersController.js
const userService = require('../services/userService');

async function registerUser(req, res) {
  try {
    const { name, location, encoding } = req.body;
    const result = await userService.registerUser(name, location, encoding);
    res.status(201).json(result);
  } catch (err) {
    console.error('Error in controller:', err);
    res.status(500).json({ error: 'Error registering user' });
  }
}

module.exports = { registerUser };