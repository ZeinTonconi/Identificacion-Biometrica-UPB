// services/userService.js
const userRepository = require('../repositories/userRepository');

async function registerUser(name, location, encoding) {
  try {
    const user = await userRepository.createUser(name, location);
    const face = await userRepository.createFace(user.user_id, encoding);
    return { user, face };
  } catch (err) {
    throw new Error(`Error in service: ${err.message}`);
  }
}

module.exports = { registerUser };