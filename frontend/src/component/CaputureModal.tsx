import { useEffect, useRef, useState } from "react";

type CaptureModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (image: string) => void;
};

const CaptureModal = ({ isOpen, onClose, onCapture }: CaptureModalProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    if (isOpen) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((mediaStream) => {
          setStream(mediaStream);
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream;
          }
        })
        .catch((err) => console.error("Error al acceder a la cámara:", err));
    } else {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    }
  }, [isOpen]);

  const takePhoto = () => {
    if (!canvasRef.current || !videoRef.current) return;
    const context = canvasRef.current.getContext("2d");
    if (!context) return;
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    context.drawImage(videoRef.current, 0, 0);
    const imageData = canvasRef.current.toDataURL("image/png");
    onCapture(imageData);
    onClose();
  };

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
        <canvas ref={canvasRef} className="hidden" />
        <div className="flex gap-4 mt-6">
          <button
            onClick={takePhoto}
            className="flex items-center gap-2 bg-blue-600 text-white px-5 py-2 rounded-xl shadow hover:bg-blue-700 transition"
          >
            Capturar
          </button>
          <button
            onClick={onClose}
            className="bg-gray-500 text-white px-5 py-2 rounded-xl shadow hover:bg-gray-600 transition"
          >
            Cerrar
          </button>
        </div>
      </div>
    </div>
  );
};

export default CaptureModal;

