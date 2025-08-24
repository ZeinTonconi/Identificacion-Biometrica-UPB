-- CreateTable
CREATE TABLE "public"."user_types" (
    "user_type_id" SERIAL NOT NULL,
    "type_name" TEXT NOT NULL,

    CONSTRAINT "user_types_pkey" PRIMARY KEY ("user_type_id")
);

-- CreateTable
CREATE TABLE "public"."users" (
    "user_id" SERIAL NOT NULL,
    "ci" TEXT NOT NULL,
    "first_name" TEXT NOT NULL,
    "last_name" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "phone" TEXT,
    "user_type_id" INTEGER NOT NULL,
    "code" INTEGER NOT NULL,
    "status" BOOLEAN NOT NULL DEFAULT true,
    "registration_date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "users_pkey" PRIMARY KEY ("user_id")
);

-- CreateTable
CREATE TABLE "public"."faces" (
    "face_id" SERIAL NOT NULL,
    "user_id" INTEGER NOT NULL,
    "encoding" TEXT NOT NULL,
    "image_path" TEXT,
    "upload_date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "faces_pkey" PRIMARY KEY ("face_id")
);

-- CreateTable
CREATE TABLE "public"."devices" (
    "device_id" SERIAL NOT NULL,
    "name" TEXT NOT NULL,
    "location" TEXT,
    "ip_address" TEXT,
    "status" BOOLEAN NOT NULL DEFAULT true,
    "registration_date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "devices_pkey" PRIMARY KEY ("device_id")
);

-- CreateTable
CREATE TABLE "public"."access_logs" (
    "log_id" SERIAL NOT NULL,
    "user_id" INTEGER,
    "device_id" INTEGER NOT NULL,
    "access_date" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "confidence" DOUBLE PRECISION NOT NULL,
    "access_type" TEXT,
    "status" TEXT,
    "enterCode" BOOLEAN,

    CONSTRAINT "access_logs_pkey" PRIMARY KEY ("log_id")
);

-- CreateIndex
CREATE UNIQUE INDEX "user_types_type_name_key" ON "public"."user_types"("type_name");

-- CreateIndex
CREATE UNIQUE INDEX "users_ci_key" ON "public"."users"("ci");

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "public"."users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "users_code_key" ON "public"."users"("code");

-- AddForeignKey
ALTER TABLE "public"."users" ADD CONSTRAINT "users_user_type_id_fkey" FOREIGN KEY ("user_type_id") REFERENCES "public"."user_types"("user_type_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."faces" ADD CONSTRAINT "faces_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."access_logs" ADD CONSTRAINT "access_logs_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."users"("user_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."access_logs" ADD CONSTRAINT "access_logs_device_id_fkey" FOREIGN KEY ("device_id") REFERENCES "public"."devices"("device_id") ON DELETE RESTRICT ON UPDATE CASCADE;
