const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function main() {
  // Seed user_types
  await prisma.user_types.createMany({
    data: [
      { type_name: 'Docente' },
      { type_name: 'Administrativo' },
    ],
    skipDuplicates: true,
  });

  // Seed users
  await prisma.users.createMany({
    data: [
      {
        ci: '12345678',
        first_name: 'Juan',
        last_name: 'Pérez',
        email: 'juan.perez@example.com',
        phone: '123456789',
        user_type_id: 1,
        code: 1001,
        status: true,
      },
      {
        ci: '87654321',
        first_name: 'María',
        last_name: 'Gómez',
        email: 'maria.gomez@example.com',
        phone: '987654321',
        user_type_id: 2,
        code: 1002,
        status: true,
      },
    ],
    skipDuplicates: true,
  });

  // Seed faces
  await prisma.faces.createMany({
    data: [
      { user_id: 1, encoding: 'encoding_data_1', image_path: '/images/juan.jpg' },
      { user_id: 2, encoding: 'encoding_data_2', image_path: '/images/maria.jpg' },
    ],
    skipDuplicates: true,
  });

  // Seed devices
  await prisma.devices.createMany({
    data: [
      { name: 'Camera1', location: 'Entrada Principal', ip_address: '192.168.1.101', status: true },
      { name: 'Camera2', location: 'Biblioteca', ip_address: '192.168.1.102', status: true },
    ],
    skipDuplicates: true,
  });

  // Seed access_logs
  await prisma.access_logs.createMany({
    data: [
      { user_id: 1, device_id: 1, confidence: 95.50, access_type: 'entrada', status: 'reconocido', enterCode: false },
      { user_id: 2, device_id: 2, confidence: 88.75, access_type: 'salida', status: 'reconocido', enterCode: false },
    ],
    skipDuplicates: true,
  });

  console.log('Seeding completed successfully');
}

main()
  .catch((e) => {
    console.error('Error during seeding:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });