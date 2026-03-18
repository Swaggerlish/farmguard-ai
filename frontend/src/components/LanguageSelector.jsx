import React from "react";

export default function LanguageSelector({ language, onLanguageChange }) {
  const languages = [
    { code: "english", name: "English", flag: "🇺🇸" },
    { code: "pidgin", name: "Pidgin", flag: "🇳🇬" },
    { code: "yoruba", name: "Yoruba", flag: "🇳🇬" },
  ];

  return (
    <div className="language-selector">
      <label htmlFor="language-select">Language:</label>
      <select
        id="language-select"
        value={language}
        onChange={(e) => onLanguageChange(e.target.value)}
        className="language-dropdown"
      >
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.flag} {lang.name}
          </option>
        ))}
      </select>
    </div>
  );
}