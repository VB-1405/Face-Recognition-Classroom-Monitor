import React, { useState, useRef, useEffect } from 'react';
import { Camera, Users, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

export default function ClassroomMonitor() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [currentMode, setCurrentMode] = useState('idle'); // idle, enrollment, monitoring
  const [students, setStudents] = useState([]);
  const [enrollmentName, setEnrollmentName] = useState('');
  const [alerts, setAlerts] = useState([]);
  const [cameraActive, setCameraActive] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const faceDetectionInterval = useRef(null);
  const mouthMovementInterval = useRef(null);

  // Initialize camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
      }
    } catch (err) {
      addAlert('Camera access denied. Please allow camera permissions.', 'error');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setCameraActive(false);
    }
  };

  // Add alert with auto-dismiss
  const addAlert = (message, type = 'info') => {
    const newAlert = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setAlerts(prev => [newAlert, ...prev].slice(0, 10));
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      setAlerts(prev => prev.filter(a => a.id !== newAlert.id));
    }, 5000);
  };

  // Simulate face detection (in production, you'd use a library like face-api.js or MediaPipe)
  const detectFaces = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0);
    
    // This is a placeholder - in production you'd use actual face detection
    // For demo purposes, we'll simulate detection
    return Math.random() > 0.3; // Simulates face detected 70% of the time
  };

  // Simulate mouth movement detection
  const detectMouthMovement = () => {
    // In production, this would analyze facial landmarks
    // For now, we'll simulate random movement detection
    return Math.random() > 0.85; // 15% chance of detecting movement
  };

  // Start enrollment mode
  const startEnrollment = async () => {
    if (!enrollmentName.trim()) {
      addAlert('Please enter a student name first', 'error');
      return;
    }
    
    setCurrentMode('enrollment');
    await startCamera();
    
    // Simulate enrollment process
    setTimeout(() => {
      const newStudent = {
        id: Date.now(),
        name: enrollmentName,
        enrolledAt: new Date().toISOString(),
        faceData: `face_${enrollmentName}_${Date.now()}` // Simulated face encoding
      };
      
      setStudents(prev => [...prev, newStudent]);
      addAlert(`${enrollmentName} enrolled successfully!`, 'success');
      setEnrollmentName('');
      setCurrentMode('idle');
      stopCamera();
    }, 3000);
  };

  // Start monitoring mode
  const startMonitoring = async () => {
    if (students.length === 0) {
      addAlert('Please enroll at least one student first', 'error');
      return;
    }
    
    setCurrentMode('monitoring');
    setIsMonitoring(true);
    await startCamera();
    
    // Start face detection loop
    faceDetectionInterval.current = setInterval(() => {
      detectFaces();
    }, 1000);
    
    // Start mouth movement detection loop
    mouthMovementInterval.current = setInterval(() => {
      if (detectMouthMovement() && students.length > 0) {
        const randomStudent = students[Math.floor(Math.random() * students.length)];
        addAlert(`${randomStudent.name}, please keep quiet during the test. Thank you!`, 'warning');
      }
    }, 3000);
  };

  // Stop monitoring
  const stopMonitoring = () => {
    setIsMonitoring(false);
    setCurrentMode('idle');
    stopCamera();
    
    if (faceDetectionInterval.current) {
      clearInterval(faceDetectionInterval.current);
    }
    if (mouthMovementInterval.current) {
      clearInterval(mouthMovementInterval.current);
    }
  };

  // Delete student
  const deleteStudent = (id) => {
    setStudents(prev => prev.filter(s => s.id !== id));
    addAlert('Student removed from system', 'info');
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (faceDetectionInterval.current) clearInterval(faceDetectionInterval.current);
      if (mouthMovementInterval.current) clearInterval(mouthMovementInterval.current);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <Camera className="w-8 h-8 text-indigo-600" />
              <h1 className="text-3xl font-bold text-gray-800">Classroom Quiet Monitor</h1>
            </div>
            <div className={`px-4 py-2 rounded-full ${cameraActive ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>
              {cameraActive ? '● Camera Active' : '○ Camera Inactive'}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Panel - Camera and Controls */}
            <div className="space-y-4">
              <div className="bg-gray-900 rounded-lg overflow-hidden aspect-video relative">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                <canvas ref={canvasRef} className="hidden" />
                
                {!cameraActive && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                    <Camera className="w-16 h-16 text-gray-600" />
                  </div>
                )}
                
                {currentMode === 'enrollment' && (
                  <div className="absolute top-4 left-4 bg-blue-600 text-white px-4 py-2 rounded-lg">
                    Enrolling: {enrollmentName}
                  </div>
                )}
                
                {currentMode === 'monitoring' && (
                  <div className="absolute top-4 left-4 bg-red-600 text-white px-4 py-2 rounded-lg animate-pulse">
                    ● MONITORING ACTIVE
                  </div>
                )}
              </div>

              {/* Enrollment Section */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-lg mb-3 text-gray-800">Student Enrollment</h3>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={enrollmentName}
                    onChange={(e) => setEnrollmentName(e.target.value)}
                    placeholder="Enter student name"
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    disabled={currentMode !== 'idle'}
                  />
                  <button
                    onClick={startEnrollment}
                    disabled={currentMode !== 'idle' || !enrollmentName.trim()}
                    className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
                  >
                    Enroll
                  </button>
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Students only need to enroll once. Their data is saved permanently.
                </p>
              </div>

              {/* Monitoring Controls */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-semibold text-lg mb-3 text-gray-800">Test Monitoring</h3>
                {!isMonitoring ? (
                  <button
                    onClick={startMonitoring}
                    disabled={currentMode !== 'idle'}
                    className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition font-semibold"
                  >
                    Start Monitoring
                  </button>
                ) : (
                  <button
                    onClick={stopMonitoring}
                    className="w-full px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition font-semibold"
                  >
                    Stop Monitoring
                  </button>
                )}
              </div>
            </div>

            {/* Right Panel - Students and Alerts */}
            <div className="space-y-4">
              {/* Enrolled Students */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <Users className="w-5 h-5 text-indigo-600" />
                  <h3 className="font-semibold text-lg text-gray-800">
                    Enrolled Students ({students.length})
                  </h3>
                </div>
                
                {students.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">No students enrolled yet</p>
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {students.map(student => (
                      <div key={student.id} className="flex items-center justify-between bg-white p-3 rounded-lg shadow-sm">
                        <div>
                          <p className="font-medium text-gray-800">{student.name}</p>
                          <p className="text-xs text-gray-500">
                            Enrolled: {new Date(student.enrolledAt).toLocaleDateString()}
                          </p>
                        </div>
                        <button
                          onClick={() => deleteStudent(student.id)}
                          className="text-red-500 hover:text-red-700 transition"
                        >
                          <XCircle className="w-5 h-5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Alert Log */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <AlertCircle className="w-5 h-5 text-orange-600" />
                  <h3 className="font-semibold text-lg text-gray-800">Activity Log</h3>
                </div>
                
                {alerts.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">No alerts yet</p>
                ) : (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {alerts.map(alert => (
                      <div
                        key={alert.id}
                        className={`p-3 rounded-lg ${
                          alert.type === 'error' ? 'bg-red-100 text-red-800' :
                          alert.type === 'warning' ? 'bg-orange-100 text-orange-800' :
                          alert.type === 'success' ? 'bg-green-100 text-green-800' :
                          'bg-blue-100 text-blue-800'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          {alert.type === 'error' && <XCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                          {alert.type === 'success' && <CheckCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                          {alert.type === 'warning' && <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />}
                          <div className="flex-1">
                            <p className="text-sm font-medium">{alert.message}</p>
                            <p className="text-xs opacity-75">{alert.timestamp}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">How to Use</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-indigo-50 p-4 rounded-lg">
              <div className="font-semibold text-indigo-900 mb-2">1. Enroll Students</div>
              <p className="text-sm text-gray-700">
                Enter each student's name and click "Enroll". They'll look at the camera for 3 seconds. This only needs to be done once.
              </p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="font-semibold text-green-900 mb-2">2. Start Monitoring</div>
              <p className="text-sm text-gray-700">
                Click "Start Monitoring" when the test begins. The system will watch for mouth movement.
              </p>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <div className="font-semibold text-orange-900 mb-2">3. Automatic Reminders</div>
              <p className="text-sm text-gray-700">
                When a student talks, they'll receive a polite reminder to stay quiet during the test.
              </p>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>Note:</strong> This is a demonstration version. For production use on macOS, you'll need to integrate actual face detection libraries like MediaPipe or face-api.js, and implement proper data storage. The current version simulates the detection for demonstration purposes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
