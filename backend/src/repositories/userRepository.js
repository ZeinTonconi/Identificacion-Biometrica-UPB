// repositories/userRepository.js
const prisma = require('../db');

async function createUser(name, location) {
  const userType = await prisma.user_types.findFirst({ where: { type_name: 'Docente' } });
  if (!userType) {
    throw new Error('User type "Docente" not found');
  }
  const user = await prisma.users.create({
    data: {
      first_name: name,
      last_name: '',  // Puedes ajustar si necesitas un apellido
      email: `${name.toLowerCase().replace(/\s+/g, '.')}@example.com`,  // Genera un email único
      ci: `AUTO_${Date.now()}`,
      phone: null,
      user_type_id: userType.user_type_id,
      code: Math.floor(1000 + Math.random() * 9000),
      status: true,
    },
  });
  return user;
}

async function createFace(userId, encoding) {
  const face = await prisma.faces.create({
    data: {
      user_id: userId,
      encoding: encoding,
      image_path: null,  // Puedes agregar un path si lo necesitas
    },
  });
  return face;
}

module.exports = { createUser, createFace };