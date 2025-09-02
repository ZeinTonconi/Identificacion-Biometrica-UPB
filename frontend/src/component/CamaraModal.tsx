import { useEffect, useRef } from "react";

interface CamaraModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const CamaraModal: React.FC<CamaraModalProps> = ({ isOpen, onClose }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let timeout: number;
    let stream: MediaStream | null = null;

    if (isOpen && videoRef.current) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((s) => {
          stream = s;
          if (videoRef.current) {
            videoRef.current.srcObject = s;
          }
          timeout = window.setTimeout(() => {
            if (stream) {
              stream.getTracks().forEach(track => track.stop());
            }
            onClose();
          }, 5000);
        })
        .catch((err) => {
          console.error("Error accediendo a la cámara:", err);
        });
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      clearTimeout(timeout);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

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
        <p className="mt-2 text-sm text-gray-500">
          Se cerrará automáticamente en 5 segundos...
        </p>
      </div>
    </div>
  );
};

export default CamaraModal;
