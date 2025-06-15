'use client';

import { useEffect, useState, useRef } from 'react';
import axios from 'axios';

interface Point { x: number; y: number; }

interface ChessLiveProps {
  mode?: 'video' | 'details';
  calibrating?: boolean;
  calibrationPoints?: Point[];
  calibrationMsg?: string;
  onCalibrate?: () => void;
  onImgClick?: (x: number, y: number) => void;
  onFenUpdate?: (fen: string) => void;
}

export default function ChessLive({ mode, calibrating: propCalibrating, calibrationPoints: propCalibrationPoints, calibrationMsg: propCalibrationMsg, onCalibrate, onImgClick, onFenUpdate }: ChessLiveProps) {
  const [fen, setFen] = useState('');
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [lastMove, setLastMove] = useState('');
  // Internal state for standalone mode
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([]);
  const [calibrationMsg, setCalibrationMsg] = useState('');
  const [forceMsg, setForceMsg] = useState('');
  const imgRef = useRef<HTMLImageElement>(null);

  // Use Next.js API proxy endpoints
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('/api/fen').then(res => {
        setFen(res.data.fen);
        setMoveHistory(res.data.move_history);
        setLastMove(res.data.last_move);
        if (onFenUpdate) onFenUpdate(res.data.fen);
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [onFenUpdate]);

  // Use props if provided, otherwise internal state
  const isCalibrating = propCalibrating !== undefined ? propCalibrating : calibrating;
  const points = propCalibrationPoints !== undefined ? propCalibrationPoints : calibrationPoints;
  const msg = propCalibrationMsg !== undefined ? propCalibrationMsg : calibrationMsg;

  const handleReset = () => {
    axios.post('/api/control', { action: 'reset' });
  };

  // For details section
  const handleCalibrate = () => {
    if (onCalibrate) return onCalibrate();
    setCalibrating(true);
    setCalibrationPoints([]);
    setCalibrationMsg('Click the 4 corners of the chessboard (top-left, top-right, bottom-right, bottom-left)');
  };

  // For video section
  const handleImgClick = (e: React.MouseEvent<HTMLImageElement, MouseEvent>) => {
    if (onImgClick) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * e.currentTarget.naturalWidth;
      const y = ((e.clientY - rect.top) / rect.height) * e.currentTarget.naturalHeight;
      onImgClick(x, y);
      return;
    }
    if (!calibrating || calibrationPoints.length >= 4) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * e.currentTarget.naturalWidth;
    const y = ((e.clientY - rect.top) / rect.height) * e.currentTarget.naturalHeight;
    setCalibrationPoints(points => {
      const newPoints = [...points, { x, y }];
      if (newPoints.length === 4) {
        // Send to backend
        axios.post('/api/calibrate', { corners: newPoints }).then(() => {
          setCalibrationMsg('Calibration complete!');
          setTimeout(() => {
            setCalibrating(false);
            setCalibrationMsg('');
          }, 1500);
        }).catch(() => {
          setCalibrationMsg('Calibration failed. Try again.');
        });
      }
      return newPoints;
    });
  };

  const handleForceStart = () => {
    axios.post('/api/force_start').then(() => {
      setForceMsg('Game force started!');
      setTimeout(() => setForceMsg(''), 2000);
    });
  };

  // Video/calibration UI
  const videoSection = (
    <div style={{ position: 'relative', width: 1800 }}>
      <img
        ref={imgRef}
        src="/api/video_feed"
        alt="Chess Video"
        style={{ width: 1800, border: '1px solid #ccc', borderRadius: 12, cursor: isCalibrating ? 'crosshair' : 'default', boxShadow: '0 2px 16px rgba(0,0,0,0.07)' }}
        onClick={handleImgClick}
      />
      {/* Draw calibration points */}
      {isCalibrating && points.map((pt, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: `${(pt.x / (imgRef.current?.naturalWidth || 640)) * 100}%`,
            top: `${(pt.y / (imgRef.current?.naturalHeight || 480)) * 100}%`,
            width: 16,
            height: 16,
            background: 'red',
            borderRadius: '50%',
            border: '2px solid white',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            zIndex: 2,
          }}
        >
          <span style={{ color: 'white', fontWeight: 'bold', fontSize: 12, position: 'absolute', left: 18, top: -8 }}>{i+1}</span>
        </div>
      ))}
      {isCalibrating && msg && (
        <div style={{ position: 'absolute', top: 8, left: 8, background: 'rgba(0,0,0,0.7)', color: 'white', padding: 8, borderRadius: 4, zIndex: 3 }}>
          {msg}
        </div>
      )}
    </div>
  );

  // Details/controls UI
  const detailsSection = (
    <div style={{ width: '100%' }}>
      <h2 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: 12, letterSpacing: '0.02em' }}>Current FEN</h2>
      <pre style={{ background: '#f7f9fb', borderRadius: 8, padding: 12, fontSize: 14, marginBottom: 18, wordBreak: 'break-all' }}>{fen}</pre>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8 }}>Last Move</h2>
      <div style={{ marginBottom: 18 }}>{lastMove}</div>
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8 }}>Move History</h2>
      <ol style={{ marginBottom: 18, paddingLeft: 18 }}>
        {moveHistory.map((move, i) => (
          <li key={i} style={{ marginBottom: 2 }}>{move}</li>
        ))}
      </ol>
      <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
        <button onClick={handleReset} disabled={isCalibrating} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: '#e6eaf0', color: '#222', fontWeight: 600, cursor: 'pointer', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>Reset Game</button>
        <button onClick={handleCalibrate} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: '#dbeafe', color: '#1e3a8a', fontWeight: 600, cursor: 'pointer', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }} disabled={isCalibrating}>Calibrate</button>
        <button onClick={handleForceStart} style={{ padding: '8px 18px', borderRadius: 8, border: 'none', background: '#bbf7d0', color: '#166534', fontWeight: 600, cursor: 'pointer', boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>Force Start</button>
      </div>
      {forceMsg && <div style={{ color: '#166534', fontWeight: 600, marginBottom: 8 }}>{forceMsg}</div>}
    </div>
  );

  if (mode === 'video') return videoSection;
  if (mode === 'details') return detailsSection;
  // fallback: both (standalone)
  return (
    <div style={{ display: 'flex', gap: 32, padding: 32 }}>
      {videoSection}
      {detailsSection}
    </div>
  );
}
