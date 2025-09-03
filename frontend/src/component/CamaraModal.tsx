import { useEffect, useRef, useState } from "react";

interface CamaraModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const CamaraModal: React.FC<CamaraModalProps> = ({ isOpen, onClose }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [name, setName] = useState<string>("Esperando detección...");

  useEffect(() => {
    let stream: MediaStream | null = null;
    let intervalId: number;

    if (isOpen && videoRef.current) {
      // Encender cámara
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = s;
          }

          intervalId = setInterval(() => {
            captureAndSend();
          }, 1000);
        })
        .catch((err) => {
          console.error("Error accediendo a la cámara:", err);
          setName("Error al acceder a la cámara");
        });
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      clearInterval(intervalId);
    };
  }, [isOpen]);

  if (!isOpen) return null;

  // Función para capturar un frame y enviarlo al backend
  const captureAndSend = async () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob>((resolve) => {
      canvas.toBlob((b) => resolve(b!), "image/jpeg");
    });

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const res = await fetch("http://localhost:8000/recognize", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setName(`${data.name} (${data.confidence.toFixed(2)}%)`);
    } catch (error) {
      console.error("Error enviando al backend:", error);
      setName("Error en la detección");
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gradient-to-b from-white to-gray-100 rounded-3xl shadow-2xl p-6 w-80 md:w-[32rem] flex flex-col items-center border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">📷 Cámara</h2>

        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="rounded-xl w-full shadow-md border border-gray-300 object-cover"
        />

        <p className="mt-4 text-base font-medium text-gray-700">{name}</p>

        <div className="mt-6">
          <button
            onClick={onClose}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Cerrar
          </button>
        </div>
      </div>
    </div>
  );
};

export default CamaraModal;
