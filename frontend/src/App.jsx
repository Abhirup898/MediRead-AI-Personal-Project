import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileText, Pill, Download, RefreshCw, Eye, EyeOff, CheckCircle, AlertTriangle, Activity, Clock, Target, BarChart3 } from 'lucide-react';

const MediReadAI = () => {
  const [currentImage, setCurrentImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [showExtractedText, setShowExtractedText] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const API_BASE_URL = 'http://localhost:5000';

  // Handle file selection
  const handleFileSelect = useCallback((files) => {
    const file = files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('Image file is too large. Please select an image smaller than 10MB.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setCurrentImage(e.target.result);
      setImagePreview(e.target.result);
      setResults(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  // Handle drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files);
    }
  }, [handleFileSelect]);

  // Process prescription
  const processPrescription = async () => {
    if (!currentImage) {
      setError('Please select an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/extract`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: currentImage
        })
      });

      const result = await response.json();

      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'Failed to extract prescription information.');
      }
    } catch (err) {
      setError(`Network error: ${err.message}. Make sure the API server is running on ${API_BASE_URL}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Download results
  const downloadResults = () => {
    if (!results) return;
    
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `prescription_results_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Clear all
  const clearAll = () => {
    setCurrentImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    setShowExtractedText(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Stat Card Component
  const StatCard = ({ icon: Icon, label, value, color = "blue" }) => (
    <div className="bg-white rounded-xl p-6 shadow-lg border-l-4 border-blue-500 hover:shadow-xl transition-all duration-300">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-600 text-sm font-medium">{label}</p>
          <p className={`text-2xl font-bold text-${color}-600 mt-1`}>{value}</p>
        </div>
        <Icon className={`h-8 w-8 text-${color}-500`} />
      </div>
    </div>
  );

  // Medicine Card Component
  const MedicineCard = ({ medicine }) => {
    const confidence = (medicine.confidence * 100).toFixed(1);
    const isValidated = medicine.validated;
    
    return (
      <div className={`bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 border-l-4 ${
        isValidated ? 'border-green-500' : 'border-orange-500'
      }`}>
        <div className="flex items-start justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800">{medicine.medicine_name}</h3>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            isValidated 
              ? 'bg-green-100 text-green-800' 
              : 'bg-orange-100 text-orange-800'
          }`}>
            {isValidated ? '✅ Validated' : '⚠️ Unvalidated'}
          </span>
        </div>

        <div className="space-y-3">
          <div className="flex items-center">
            <span className="text-gray-600 font-medium w-24">Dosage:</span>
            <span className="text-gray-800">{medicine.dosage || 'Not specified'}</span>
          </div>
          
          <div className="flex items-center">
            <span className="text-gray-600 font-medium w-24">Frequency:</span>
            <span className="text-gray-800">{medicine.frequency || 'Not specified'}</span>
          </div>
          
          {medicine.instructions && (
            <div className="flex items-start">
              <span className="text-gray-600 font-medium w-24">Instructions:</span>
              <span className="text-gray-800 flex-1">{medicine.instructions}</span>
            </div>
          )}
          
          {medicine.generic_name && (
            <div className="flex items-center">
              <span className="text-gray-600 font-medium w-24">Generic:</span>
              <span className="text-gray-800">{medicine.generic_name}</span>
            </div>
          )}
        </div>

        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-600">Confidence Score</span>
            <span className="text-sm font-medium text-gray-800">{confidence}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-500 ${
                confidence >= 80 ? 'bg-green-500' : 
                confidence >= 60 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-xl">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-2 flex items-center justify-center gap-3">
              <Activity className="h-10 w-10" />
              MediRead AI
            </h1>
            <p className="text-xl opacity-90">Advanced AI-Powered Prescription Reading & Medicine Extraction</p>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8 max-w-7xl">
        {/* Upload Section */}
        <div className="mb-8">
          <div 
            className={`border-3 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer ${
              dragActive 
                ? 'border-blue-500 bg-blue-50 scale-105' 
                : 'border-gray-300 bg-gray-50 hover:border-blue-400 hover:bg-blue-50'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="h-16 w-16 text-blue-500 mx-auto mb-4" />
            <div className="text-xl font-semibold text-gray-700 mb-2">
              Drop your prescription image here
            </div>
            <div className="text-gray-500 mb-6">
              or click to browse files • Supports JPG, PNG • Max 10MB
            </div>
            <button className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-3 rounded-full font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 shadow-lg hover:shadow-xl">
              Choose File
            </button>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept="image/*"
              onChange={(e) => handleFileSelect(e.target.files)}
            />
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg mb-6">
            <div className="flex items-center">
              <AlertTriangle className="h-6 w-6 text-red-500 mr-3" />
              <div>
                <h3 className="text-red-800 font-semibold">Error</h3>
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Image Preview */}
        {imagePreview && (
          <div className="mb-8 bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <FileText className="h-6 w-6" />
              Image Preview
            </h3>
            <div className="text-center">
              <img 
                src={imagePreview} 
                alt="Prescription preview" 
                className="max-w-full max-h-96 rounded-lg shadow-lg mx-auto mb-6"
              />
              <div className="flex gap-4 justify-center">
                <button
                  onClick={processPrescription}
                  disabled={isLoading}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 text-white px-8 py-3 rounded-full font-semibold hover:from-green-700 hover:to-emerald-700 transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isLoading ? (
                    <RefreshCw className="h-5 w-5 animate-spin" />
                  ) : (
                    <Target className="h-5 w-5" />
                  )}
                  {isLoading ? 'Processing...' : 'Extract Medicines'}
                </button>
                <button
                  onClick={clearAll}
                  className="bg-gray-600 text-white px-8 py-3 rounded-full font-semibold hover:bg-gray-700 transition-all duration-300 shadow-lg hover:shadow-xl"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="bg-white rounded-2xl shadow-lg p-12 text-center mb-8">
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mb-6"></div>
              <h3 className="text-2xl font-bold text-gray-800 mb-2">🔍 Analyzing Prescription...</h3>
              <p className="text-gray-600">Please wait while our AI extracts medicine information</p>
            </div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="space-y-8">
            {/* Stats */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <BarChart3 className="h-6 w-6" />
                Extraction Results
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard
                  icon={Pill}
                  label="Medicines Found"
                  value={results.total_medicines}
                  color="blue"
                />
                <StatCard
                  icon={Target}
                  label="Confidence Score"
                  value={`${(results.confidence_score * 100).toFixed(1)}%`}
                  color="green"
                />
                <StatCard
                  icon={CheckCircle}
                  label="Validated"
                  value={results.medicines.filter(m => m.validated).length}
                  color="emerald"
                />
              </div>
            </div>

            {/* Extracted Text Toggle */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-gray-800">Extracted Text</h3>
                <button
                  onClick={() => setShowExtractedText(!showExtractedText)}
                  className="flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
                >
                  {showExtractedText ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                  {showExtractedText ? 'Hide' : 'Show'} Text
                </button>
              </div>
              {showExtractedText && (
                <div className="bg-gray-100 p-4 rounded-lg font-mono text-sm text-gray-700 whitespace-pre-wrap border-l-4 border-gray-400">
                  {results.extracted_text}
                </div>
              )}
            </div>

            {/* Medicines Grid */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Pill className="h-6 w-6" />
                Extracted Medicines
              </h3>
              
              {results.medicines.length === 0 ? (
                <div className="text-center py-12">
                  <AlertTriangle className="h-16 w-16 text-orange-500 mx-auto mb-4" />
                  <h4 className="text-xl font-semibold text-gray-800 mb-2">No medicines detected</h4>
                  <p className="text-gray-600">Please try with a clearer image or check if the prescription contains readable text.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {results.medicines.map((medicine, index) => (
                    <MedicineCard key={index} medicine={medicine} />
                  ))}
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex flex-wrap gap-4 justify-center">
              <button
                onClick={downloadResults}
                className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 py-3 rounded-full font-semibold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-xl flex items-center gap-2"
              >
                <Download className="h-5 w-5" />
                Download JSON
              </button>
              <button
                onClick={clearAll}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-8 py-3 rounded-full font-semibold hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 shadow-lg hover:shadow-xl flex items-center gap-2"
              >
                <RefreshCw className="h-5 w-5" />
                New Prescription
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="container mx-auto px-6 text-center">
          <p className="text-gray-400">
            © 2024 MediRead AI. Powered by advanced AI technology for better healthcare.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default MediReadAI;