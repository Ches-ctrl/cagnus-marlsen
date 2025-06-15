import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const response = await fetch('http://localhost:5001/video_feed');
  const headers = new Headers(response.headers);
  headers.delete('access-control-allow-origin');
  headers.delete('access-control-allow-credentials');
  return new Response(response.body, {
    status: response.status,
    headers,
  });
}
