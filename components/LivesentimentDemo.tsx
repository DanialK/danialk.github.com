'use client'
import { ReactNode, useState } from 'react'
import '@/css/livesentiment.css'

export default function LivesentimentDemo() {
  const [show, setShow] = useState(false)

  if (!show) {
    return (
      <div id="livesentiment-iframe-holder">
        <button id="livesentiment-start" onClick={() => setShow(true)}>
          Start
        </button>
      </div>
    )
  }

  return (
    <div id="livesentiment-iframe-holder">
      <iframe
        title="Livesentiment Demo"
        id="livesentiment-iframe"
        src="/static/livesentiment/index.html"
      ></iframe>
    </div>
  )
}
