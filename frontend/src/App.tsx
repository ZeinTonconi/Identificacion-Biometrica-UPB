import { useState } from "react";
import BuhoComponent from "./component/BuhoComponent";
import UpbLogo from "./assets/upb.png";
import CamaraModal from "./component/CamaraModal";
import CaptureModal from "./component/CaputureModal";
import SpeechReader from "./component/SpeechReader";

function App() {
  const [openCam, setOpenCam] = useState(false);
  const [openCapture, setOpenCapture] = useState(false);
  const [showSpeech, setShowSpeech] = useState(false);

  const handleCloseCam = () => {
    setOpenCam(false);
    setShowSpeech(true);
  };

  const handleCapturePhoto = (image: string) => {
    console.log("Foto de registro:", image);
  };

  return (
    <div className="bg-[#ffffff] min-h-screen flex flex-col md:flex-row">
      <BuhoComponent />
      <div
        className="
          w-full md:w-1/2
          flex flex-col justify-center items-center p-8
          shadow-lg
          rounded-t-[4.5rem] md:rounded-t-none md:rounded-l-[4.5rem]
          grow
        "
        style={{ backgroundColor: "#160660" }}
      >
        <div className="text-center space-y-6 w-full max-w-md">
          <img src={UpbLogo} alt="Upb" className="mx-auto h-16" />

          <p className="text-base sm:text-lg mt-4 text-white font-semibold">
            Bienvenido a la plataforma de identificación biométrica de la UPB
          </p>

          <div className="mt-8 space-y-4 flex flex-col items-center">
            <button
              onClick={() => {
                setOpenCam(true);
                setShowSpeech(false);
              }}
              className="bg-white py-3 px-8 text-base sm:text-lg hover:bg-[#56c8eb] transition w-full rounded-sm font-bold"
            >
              Inicio de identificación biométrica
            </button>

            <button
              onClick={() => setOpenCapture(true)}
              className="bg-[#f9b92d] py-3 px-8 text-white text-base sm:text-lg hover:bg-[#56c8eb] transition w-full rounded-sm font-bold"
            >
              Registrar nuevo usuario
            </button>
          </div>
        </div>
      </div>

      <CamaraModal isOpen={openCam} onClose={handleCloseCam} />
      <CaptureModal
        isOpen={openCapture}
        onClose={() => setOpenCapture(false)}
        onCapture={handleCapturePhoto}
      />

      {/* {showSpeech && (
        <SpeechReader text="HOLA LUIS DE LA UNIVERSIDAD PRIVADA BOLIVIANA, BIENVENIDO NUEVAMENTE A LA UNIVERSIDAD" />
      )} */}
    </div>
  );
}

export default App;
