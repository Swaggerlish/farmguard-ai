import React, { useState, useEffect } from "react";

const translations = {
  english: {
    predictionResults: "Prediction Results",
    topPredictions: "Top 3 Predictions:",
    crop: "Crop:",
    status: "Status:",
    healthy: "Healthy",
    needsAttention: "Needs attention",
    description: "Description",
    treatment: "Treatment",
    prevention: "Prevention",
    urgency: "Urgency",
    speak: "🔊 Speak",
    stop: "⏹️ Stop",
    voiceNotSupported: "Voice not supported in this browser",
  },
  pidgin: {
    predictionResults: "Prediction Results",
    topPredictions: "Top 3 Predictions:",
    crop: "Crop:",
    status: "Status:",
    healthy: "Healthy",
    needsAttention: "Needs attention",
    description: "Description",
    treatment: "Treatment",
    prevention: "Prevention",
    urgency: "Urgency",
    speak: "🔊 Tok",
    stop: "⏹️ Stop",
    voiceNotSupported: "Voice no support for this browser",
  },
  yoruba: {
    predictionResults: "Awọn Abajade Asọtẹlẹ",
    topPredictions: "Awọn Asọtẹlẹ 3 Ti O Ga Julọ:",
    crop: "Iru Irugbin:",
    status: "Ipo:",
    healthy: "Ni Ilera",
    needsAttention: "Nilo Ifarabalẹ",
    description: "Apejuwe",
    treatment: "Itọju",
    prevention: "Idena",
    urgency: "Ira",
    speak: "🔊 Sọ",
    stop: "⏹️ Duro",
    voiceNotSupported: "Ohùn ko ni atilẹyin ninu ẹrọ aṣàwákiri yii",
  },
};

export default function PredictionCard({ result, language = "english" }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);

  useEffect(() => {
    // Check if speech synthesis is supported
    if ('speechSynthesis' in window) {
      setSpeechSupported(true);
    }
  }, []);

  if (!result) return null;

  const t = translations[language] || translations.english;

  const speakText = (text) => {
    if (!speechSupported) return;

    // Stop any current speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    
    // Set language based on selected language
    if (language === 'pidgin') {
      utterance.lang = 'en-GB'; // Use English voice for Pidgin
    } else if (language === 'yoruba') {
      utterance.lang = 'en-GB'; // Use English voice for Yoruba (can be improved with local voices)
    } else {
      utterance.lang = 'en-US';
    }

    utterance.rate = 0.8; // Slightly slower for clarity
    utterance.pitch = 1;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const stopSpeech = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  const speakResults = () => {
    if (isSpeaking) {
      stopSpeech();
      return;
    }

    let textToSpeak = `${t.predictionResults}. `;

    // Speak top prediction
    const topPred = result.predictions[0];
    const percent = (topPred.confidence * 100).toFixed(0);
    textToSpeak += `Top prediction: ${topPred.disease_name} with ${percent} percent confidence. `;
    textToSpeak += `${t.crop} ${topPred.crop}. `;
    textToSpeak += `${t.status} ${topPred.healthy ? t.healthy : t.needsAttention}. `;

    // Speak advice
    textToSpeak += `${t.description}: ${result.advice.description}. `;
    textToSpeak += `${t.treatment}: ${result.advice.treatment.replace(/\n/g, '. ')}. `;
    textToSpeak += `${t.prevention}: ${result.advice.prevention}. `;
    textToSpeak += `${t.urgency}: ${result.advice.urgency}.`;

    speakText(textToSpeak);
  };

  return (
    <div className="card result-card">
      <div className="card-header">
        <h2>{t.predictionResults}</h2>
        {speechSupported && (
          <button 
            onClick={speakResults}
            className={`voice-button ${isSpeaking ? 'speaking' : ''}`}
            title={isSpeaking ? t.stop : t.speak}
          >
            {isSpeaking ? t.stop : t.speak}
          </button>
        )}
        {!speechSupported && (
          <span className="voice-notice" title={t.voiceNotSupported}>
            🔇
          </span>
        )}
      </div>
      
      <div className="predictions-list">
        <h3>{t.topPredictions}</h3>
        {result.predictions.map((pred, index) => {
          const percent = (pred.confidence * 100).toFixed(2);
          return (
            <div key={index} className={`prediction-item ${index === 0 ? 'top-prediction' : ''}`}>
              <div className="prediction-header">
                <span className="rank">#{index + 1}</span>
                <span className="disease">{pred.disease_name}</span>
                <span className="confidence">{percent}%</span>
              </div>
              <div className="prediction-details">
                <span><strong>{t.crop}</strong> {pred.crop}</span>
                <span><strong>{t.status}</strong> {pred.healthy ? t.healthy : t.needsAttention}</span>
              </div>
            </div>
          );
        })}
      </div>

      {result.note && (
        <div className="note">
          <strong>Note:</strong> {result.note}
        </div>
      )}

      <div className="advice-box">
        <h3>{t.description}</h3>
        <p>{result.advice.description}</p>

        <h3>{t.treatment}</h3>
        <ul className="treatment-list">
          {result.advice.treatment.split('\n').filter(item => item.trim()).map((item, index) => (
            <li key={index}>{item.trim()}</li>
          ))}
        </ul>

        <h3>{t.prevention}</h3>
        <p>{result.advice.prevention}</p>

        <h3>{t.urgency}</h3>
        <p>{result.advice.urgency}</p>
      </div>
    </div>
  );
}