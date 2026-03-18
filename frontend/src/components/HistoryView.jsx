import React from "react";

export default function HistoryView({ history, onViewScan, onClearHistory, language, translations }) {
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getTopPrediction = (result) => {
    if (!result || !result.predictions || result.predictions.length === 0) return "No prediction";
    return result.predictions[0].disease_name;
  };

  // Add view details translation
  const viewDetailsText = {
    english: "View Details",
    pidgin: "See Details",
    yoruba: "Wo Awọn Ẹkunrin",
  };

  if (history.length === 0) {
    return (
      <div className="history-empty">
        <div className="empty-state">
          <h3>{translations.noHistory}</h3>
        </div>
      </div>
    );
  }

  return (
    <div className="history-view">
      <div className="history-header">
        <h2>Scan History</h2>
        <button 
          className="clear-history-btn"
          onClick={onClearHistory}
          disabled={history.length === 0}
        >
          {translations.clearHistory}
        </button>
      </div>

      <div className="history-list">
        {history.map((scan) => (
          <div key={scan.id} className="history-item">
            <div className="history-item-content">
              <div className="history-item-header">
                <span className="scan-date">{formatDate(scan.timestamp)}</span>
                <span className="scan-language">{scan.language.toUpperCase()}</span>
              </div>
              <div className="history-item-details">
                <span className="scan-prediction">{getTopPrediction(scan.result)}</span>
                <span className="scan-image">{scan.imageName}</span>
              </div>
            </div>
            <button 
              className="view-scan-btn"
              onClick={() => onViewScan(scan)}
            >
              {viewDetailsText[language] || viewDetailsText.english}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}