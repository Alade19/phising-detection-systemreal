import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Phishing URL Detector",
  description: "AI-powered phishing URL detection tool",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-gray-100">{children}</body>
    </html>
  );
}
