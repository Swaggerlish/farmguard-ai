import React, { useState } from "react";

export default function UploadForm({ onSubmit, loading, language }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleChange = (e) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onSubmit(file);
  };

  return (
    <div className="card">
      <h2>{language === 'pidgin' ? 'Upload Lif Imej' :
           language === 'yoruba' ? 'Gbe Aworan Ewe Si' :
           'Upload Leaf Image'}</h2>
      <form onSubmit={handleSubmit}>
        <div className="file-input-group">
          <label htmlFor="leaf-image" className="file-input-label">
            {language === 'pidgin' ? 'Pik wan klia lif imej wey yu want mek ai analiz' :
             language === 'yoruba' ? 'Yan aworan ewe ti o fẹ ki ai ṣe ayẹwo' :
             'Choose a clear leaf image for analysis'}
          </label>
          <input 
            type="file" 
            id="leaf-image"
            accept="image/*" 
            onChange={handleChange}
            className="file-input"
          />
        </div>
        {preview && (
          <div className="preview-wrap">
            <img src={preview} alt="Leaf preview" className="preview-image" />
          </div>
        )}
        <button type="submit" disabled={!file || loading}>
          {loading ? (language === 'pidgin' ? 'Ai de analiz...' :
                      language === 'yoruba' ? 'Ai n ṣe ayẹwo...' :
                      'Analyzing...') :
           language === 'pidgin' ? 'Analiz di lif' :
           language === 'yoruba' ? 'Ṣe ayẹwo ewe' :
           'Analyze Image'}
        </button>
      </form>
    </div>
  );
}