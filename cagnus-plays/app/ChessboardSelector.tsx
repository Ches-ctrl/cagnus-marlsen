"use client";
import React, { useRef, useEffect, useState } from "react";

const VIDEO_URL = "http://localhost:5001/video_feed";

export default function ChessboardSelector() {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [corners, setCorners] = useState<{ x: number; y: number }[]>([]);

  // Draw markers on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (canvas && img) {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      corners.forEach((corner, i) => {
        ctx.beginPath();
        ctx.arc(corner.x, corner.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
        ctx.font = "16px Arial";
        ctx.fillText(`${i + 1}`, corner.x + 10, corner.y - 10);
      });
    }
  }, [corners]);

  // Handle click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * img.naturalWidth) / canvas.width;
    const y = ((e.clientY - rect.top) * img.naturalHeight) / canvas.height;
    if (corners.length < 4) {
      setCorners([...corners, { x, y }]);
    }
  };

  // Send corners to backend
  const sendCorners = async () => {
    await fetch("http://localhost:5001/set_corners", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corners }),
    });
    alert("Corners sent!");
  };

  // Reset selection
  const resetCorners = () => setCorners([]);

  return (
    <div style={{ position: "relative", width: 480 }}>
      <img
        ref={imgRef}
        src={VIDEO_URL}
        alt="Live Chessboard"
        style={{ width: 480, display: "block" }}
        onLoad={() => {
          if (canvasRef.current && imgRef.current) {
            canvasRef.current.width = imgRef.current.width;
            canvasRef.current.height = imgRef.current.height;
          }
        }}
      />
      <canvas
        ref={canvasRef}
        width={480}
        height={360}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          pointerEvents: "auto",
          width: 480,
          height: 360,
        }}
        onClick={handleCanvasClick}
      />
      <div style={{ marginTop: 8 }}>
        <button onClick={sendCorners} disabled={corners.length !== 4}>
          Send Corners
        </button>
        <button onClick={resetCorners} style={{ marginLeft: 8 }}>
          Reset
        </button>
      </div>
    </div>
  );
}
