"use client";

import { useRef, useState, useEffect } from "react";
import Image from "next/image";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [corners, setCorners] = useState<[number, number][]>([]);
  const [fen, setFen] = useState<string | null>(null);
  const [setupStep, setSetupStep] = useState<'idle' | 'started' | 'corners' | 'ready'>('idle');
  const [error, setError] = useState<string | null>(null);
  const [detectedMove, setDetectedMove] = useState<string | null>(null);
  const [engineMove, setEngineMove] = useState<string | null>(null);
  const [loopActive, setLoopActive] = useState(false);

  // Start webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch {
      setError("Could not access webcam");
    }
  };

  // Start game (get session id)
  const startGame = async () => {
    const res = await fetch("http://localhost:5001/start-game", { method: "POST" });
    const data = await res.json();
    setSessionId(data.session_id);
    setSetupStep('started');
    startWebcam();
  };

  // Capture still frame for corner selection
  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    setSetupStep('corners');
  };

  // Handle user clicking on the canvas to select corners
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (setupStep !== 'corners') return;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (corners.length < 4) {
      setCorners([...corners, [x, y]]);
    }
  };

  // Send corners to backend
  const submitCorners = async () => {
    if (!sessionId || corners.length !== 4) return;
    await fetch("http://localhost:5001/set-board-corners", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, corners }),
    });
    setSetupStep('ready');
    setLoopActive(true);
  };

  // Capture a frame and send to backend for FEN/move detection
  const sendFrame = async () => {
    if (!sessionId || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx || !videoRef.current) return;
    ctx.drawImage(videoRef.current, 0, 0, 640, 480);
    const dataUrl = canvasRef.current.toDataURL("image/jpeg");
    const base64 = dataUrl.split(",")[1];
    const res = await fetch("http://localhost:5001/upload-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, image: base64 }),
    });
    const data = await res.json();
    setFen(data.fen);
    setDetectedMove(data.detected_move || null);
    return data;
  };

  // Call engine to make white's move
  const makeEngineMove = async () => {
    if (!sessionId) return;
    const res = await fetch("http://localhost:5001/engine-move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await res.json();
    setFen(data.fen);
    setEngineMove(data.engine_move || null);
    // Placeholder: call robot agent here with data.engine_move
    // await callRobotAgent(data.engine_move);
  };

  // Main game loop
  useEffect(() => {
    let loop = true;
    async function visionLoop() {
      while (loopActive && loop) {
        const data = await sendFrame();
        if (data) {
          // If a human (black) move was detected and it's now white's turn
          if (data.detected_move && data.turn === 'w') {
            await makeEngineMove();
          }
        }
        await new Promise(res => setTimeout(res, 2000));
      }
    }
    if (loopActive) {
      loop = true;
      visionLoop();
    }
    return () => { loop = false; };
    // eslint-disable-next-line
  }, [loopActive, sessionId]);

  return (
    <div style={{ padding: 24, minHeight: '100vh', boxSizing: 'border-box', background: '#f4f7fa' }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        marginBottom: 32,
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 24,
        }}>
          <div style={{
            width: 72,
            height: 72,
            borderRadius: '50%',
            overflow: 'hidden',
            border: '4px solid #ba0c2f', // Norwegian red
            boxShadow: '0 2px 12px rgba(0,0,0,0.10)',
            background: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <Image src="/cagnus_2.jpeg" alt="Cagnus" width={72} height={72} style={{ objectFit: 'cover', width: '100%', height: '100%' }} />
          </div>
          <h1 style={{
            textAlign: 'center',
            margin: 0,
            fontSize: 32,
            fontWeight: 800,
            letterSpacing: 1,
            color: '#00205b', // Norwegian blue
            padding: '0 16px',
            borderLeft: '8px solid #fff',
            borderRight: '8px solid #ba0c2f',
            background: 'linear-gradient(90deg, #fff 0%, #fff 60%, #ba0c2f 100%)',
            borderRadius: 12,
            boxShadow: '0 2px 12px rgba(0,0,0,0.04)',
          }}>
            Cagnus Marlsen Chess Vision
          </h1>
        </div>
        <div style={{
          width: 120,
          height: 12,
          marginTop: 12,
          background: 'linear-gradient(90deg, #ba0c2f 0%, #fff 40%, #00205b 100%)',
          borderRadius: 6,
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', gap: 48, maxWidth: 1300, margin: '0 auto' }}>
        {/* Left: Video/Canvas */}
        <div style={{ flex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minWidth: 660 }}>
          {(setupStep === 'started' || setupStep === 'corners' || setupStep === 'ready') && (
            <>
              <video
                ref={videoRef}
                width={800}
                height={600}
                autoPlay
                style={{
                  display: setupStep === 'started' ? 'block' : 'none',
                  borderRadius: 18,
                  boxShadow: '0 4px 32px rgba(186,12,47,0.10), 0 0 0 6px #00205b', // blue shadow
                  background: '#222',
                  maxWidth: '100%',
                  marginBottom: 16,
                  border: '4px solid #fff',
                }}
              />
              <canvas
                ref={canvasRef}
                width={800}
                height={600}
                style={{
                  border: '4px solid #ba0c2f', // Norwegian red
                  borderRadius: 18,
                  display: setupStep === 'corners' || setupStep === 'ready' ? 'block' : 'none',
                  cursor: setupStep === 'corners' && corners.length < 4 ? 'crosshair' : 'default',
                  boxShadow: '0 4px 32px rgba(0,32,91,0.10), 0 0 0 6px #fff', // white shadow
                  background: '#222',
                  maxWidth: '100%',
                  marginBottom: 16,
                }}
                onClick={handleCanvasClick}
              />
            </>
          )}
        </div>
        {/* Right: Controls/Info */}
        <div style={{ flex: 1, minWidth: 340, maxWidth: 400, background: '#fff', borderRadius: 16, boxShadow: '0 2px 16px rgba(0,32,91,0.07)', padding: 32, display: 'flex', flexDirection: 'column', gap: 24, border: '3px solid #00205b' }}>
          {error && <div style={{ color: 'red', marginBottom: 16 }}>{error}</div>}
          {setupStep === 'idle' && (
            <button style={{ fontSize: 20, padding: '16px 32px', borderRadius: 8, background: '#ba0c2f', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 600, boxShadow: '0 2px 8px rgba(0,32,91,0.08)' }} onClick={startGame}>Start Game</button>
          )}
          {setupStep === 'started' && (
            <button style={{ fontSize: 18, padding: '12px 24px', borderRadius: 8, background: '#00205b', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 500, marginBottom: 12, boxShadow: '0 2px 8px rgba(186,12,47,0.08)' }} onClick={captureFrame}>Capture Frame for Corner Selection</button>
          )}
          {setupStep === 'corners' && (
            <div>
              <p style={{ fontWeight: 500, marginBottom: 8, color: '#00205b' }}>Click the 4 board corners in order:</p>
              <ol style={{ margin: 0, paddingLeft: 20, marginBottom: 8, color: '#ba0c2f' }}>
                <li>Top-left</li>
                <li>Top-right</li>
                <li>Bottom-right</li>
                <li>Bottom-left</li>
              </ol>
              <p style={{ marginBottom: 8 }}>Selected: <b>{corners.length}/4</b></p>
              {corners.length === 4 && <button style={{ fontSize: 16, padding: '10px 20px', borderRadius: 8, background: '#34a853', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 500 }} onClick={submitCorners}>Submit Corners & Start Game</button>}
            </div>
          )}
          {setupStep === 'ready' && (
            <div>
              <div style={{ marginBottom: 16, color: '#388e3c', fontWeight: 600 }}>Game running! Vision loop is active.</div>
              {fen && <div style={{ marginBottom: 16 }}><b>Current FEN:</b> <pre style={{ background: '#f5f5f5', padding: 8, borderRadius: 6, fontSize: 15, margin: 0 }}>{fen}</pre></div>}
              {detectedMove && <div style={{ marginBottom: 8 }}><b>Detected Human Move:</b> <span style={{ color: '#d32f2f' }}>{detectedMove}</span></div>}
              {engineMove && <div style={{ marginBottom: 8 }}><b>Engine (Robot) Move:</b> <span style={{ color: '#1976d2' }}>{engineMove}</span></div>}
              <div style={{ marginTop: 16, fontSize: 15, color: '#555' }}>White (bottom): <b>Robot</b> | Black (top): <b>Human</b></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
