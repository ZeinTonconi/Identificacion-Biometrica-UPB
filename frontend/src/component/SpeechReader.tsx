import { useEffect } from "react";

type Props = {
  text: string;
};

const SpeechReader= ({ text }: Props) => {
  useEffect(() => {
    if (text) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "es-ES";
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
    }
  }, [text]);

  return null;
};

export default SpeechReader;
