import React, { useState, useEffect } from "react";
import UploadForm from "./components/UploadForm";
import PredictionCard from "./components/PredictionCard";
import LanguageSelector from "./components/LanguageSelector";
import HistoryView from "./components/HistoryView";
import { predictDisease } from "./services/api";

const translations = {
  english: {
    title: "FarmGuard AI",
    tagline: "Detect crop diseases instantly using artificial intelligence",
    subtitle: "Upload a crop leaf image and get AI-powered disease detection with treatment and prevention advice.",
    errorMessage: "Something went wrong while analyzing the image.",
    showCurrent: "New Scan",
    showHistory: "View History",
    clearHistory: "Clear History",
    noHistory: "No scan history yet. Upload an image to get started!",
    offlineMessage: "You are currently offline. Some features may be limited.",
    onlineMessage: "Back online! All features available.",
    offlineScanMessage: "Offline mode: Your scan will be saved locally and synced when connection is restored.",
  },
  pidgin: {
    title: "FarmGuard AI",
    tagline: "Detect crop diseases instantly using artificial intelligence",
    subtitle: "Upload crop leaf picture, make AI help you detect disease with treatment and prevention advice.",
    errorMessage: "Something go wrong while we dey analyze the image.",
    showCurrent: "New Scan",
    showHistory: "Check History",
    clearHistory: "Clear History",
    noHistory: "No scan history yet. Upload picture to start!",
    offlineMessage: "You dey offline now. Some features fit limited.",
    onlineMessage: "You don come back online! All features dey available.",
    offlineScanMessage: "Offline mode: Your scan go save locally and sync when connection come back.",
  },
  yoruba: {
    title: "FarmGuard AI",
    tagline: "Detect crop diseases instantly using artificial intelligence",
    subtitle: "Gbe aworan ewe irugbin si oke ki o gba awari arun nipasẹ AI pẹlu imọran itọju ati idena.",
    errorMessage: "Nkankan lọ si ibi nigba ti a nṣe ayẹwo aworan naa.",
    showCurrent: "Ayẹwo Tuntun",
    showHistory: "Wo Itan",
    clearHistory: "Pa Itan Rẹ",
    noHistory: "Ko si itan ayẹwo sibẹsibẹ. Gbe aworan si oke lati bẹrẹ!",
    offlineMessage: "O wa ni offline lọwọlọwọ. Diẹ ninu awọn ẹya ara le ni opin.",
    onlineMessage: "Pada si ori ayelujara! Gbogbo awọn ẹya ara wa!",
    offlineScanMessage: "Ipo offline: Ayẹwo rẹ yoo wa ni ipamọ ni agbegbe ati muṣiṣẹpọ nigbati asopọ ba pada.",
  },
};

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [language, setLanguage] = useState("english");
  const [currentView, setCurrentView] = useState("scan"); // "scan" or "history"
  const [scanHistory, setScanHistory] = useState([]);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showOfflineMessage, setShowOfflineMessage] = useState(false);

  const t = translations[language] || translations.english;

  // Load scan history from localStorage on component mount
  useEffect(() => {
    const savedHistory = localStorage.getItem("farmguard-scan-history");
    if (savedHistory) {
      try {
        setScanHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error("Error loading scan history:", e);
      }
    }
  }, []);

  // Save scan history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("farmguard-scan-history", JSON.stringify(scanHistory));
  }, [scanHistory]);

  // Handle online/offline status
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowOfflineMessage(true);
      setTimeout(() => setShowOfflineMessage(false), 3000);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setShowOfflineMessage(true);
      // Don't auto-hide offline message - keep it visible until back online
    };

    // Show initial status briefly on app load
    setShowOfflineMessage(true);
    setTimeout(() => setShowOfflineMessage(false), 2000);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handlePredict = async (file) => {
    try {
      setLoading(true);
      setError("");
      setResult(null);

      if (!isOnline) {
        // Show offline message briefly
        setError(t.offlineScanMessage);
        setTimeout(() => setError(""), 5000);
      }

      const data = await predictDisease(file, language);
      setResult(data);

      // Add to history
      const scanEntry = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        language,
        result: data,
        imageName: file.name,
      };
      setScanHistory(prev => [scanEntry, ...prev].slice(0, 50)); // Keep last 50 scans
    } catch (err) {
      setError(
        err?.response?.data?.detail || t.errorMessage
      );
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = (newLanguage) => {
    setLanguage(newLanguage);
    // Clear previous results when language changes
    setResult(null);
    setError("");
  };

  const handleViewHistory = (scan) => {
    setResult(scan.result);
    setCurrentView("scan");
  };

  const handleClearHistory = () => {
    setScanHistory([]);
  };

  // Debug function to test offline banner
  const testOfflineBanner = () => {
    setIsOnline(false);
    setShowOfflineMessage(true);
  };

  const testOnlineBanner = () => {
    setIsOnline(true);
    setShowOfflineMessage(true);
    setTimeout(() => setShowOfflineMessage(false), 3000);
  };

  return (
    <main className="app">
      {showOfflineMessage && (
        <div className={`offline-banner ${isOnline ? 'online' : 'offline'}`}>
          {isOnline ? t.onlineMessage : t.offlineMessage}
        </div>
      )}

      <section className="hero">
        <div className="hero-header">
          <h1>
            <span className="title-main">{t.title}</span>
            <span className="title-separator">–</span>
            <span className="title-tagline">{t.tagline}</span>
          </h1>
          <LanguageSelector language={language} onLanguageChange={handleLanguageChange} />
        </div>
        <p>
          {t.subtitle}
        </p>
        
        <div className="view-tabs">
          <button 
            className={`tab-button ${currentView === "scan" ? "active" : ""}`}
            onClick={() => setCurrentView("scan")}
          >
            {t.showCurrent}
          </button>
          <button 
            className={`tab-button ${currentView === "history" ? "active" : ""}`}
            onClick={() => setCurrentView("history")}
          >
            {t.showHistory} ({scanHistory.length})
          </button>
        </div>
      </section>

      {currentView === "scan" ? (
        <div className="grid">
          <UploadForm onSubmit={handlePredict} loading={loading} />
          <PredictionCard result={result} language={language} />
        </div>
      ) : (
        <HistoryView 
          history={scanHistory} 
          onViewScan={handleViewHistory}
          onClearHistory={handleClearHistory}
          language={language}
          translations={t}
        />
      )}

      {error && <div className="error-box">{error}</div>}
      
      {/* Debug buttons - remove in production */}
      <div style={{ position: 'fixed', bottom: '10px', right: '10px', zIndex: 1000 }}>
        <button onClick={testOfflineBanner} style={{ marginRight: '5px', padding: '5px 10px', background: '#f59e0b' }}>Test Offline</button>
        <button onClick={testOnlineBanner} style={{ padding: '5px 10px', background: '#10b981' }}>Test Online</button>
      </div>
    </main>
  );
}