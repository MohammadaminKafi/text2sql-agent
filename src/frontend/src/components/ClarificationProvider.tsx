import { useState, useEffect } from 'react'
import { Card } from '@/components/Card'

interface ClarificationData {
  clarification_id: string
  question: string
  preset_answers: string[]
  elapsed_seconds: number
}

interface ClarificationModalProps {
  clarification: ClarificationData
  onAnswer: (answer: string, isPreset: boolean, presetIndex?: number) => void
  onCancel: () => void
}

function ClarificationModal({ clarification, onAnswer, onCancel }: ClarificationModalProps) {
  const [customAnswer, setCustomAnswer] = useState('')
  const [selectedPreset, setSelectedPreset] = useState<number | null>(null)

  const handlePresetClick = (index: number, answer: string) => {
    setSelectedPreset(index)
    onAnswer(answer, true, index)
  }

  const handleCustomSubmit = () => {
    if (customAnswer.trim()) {
      onAnswer(customAnswer.trim(), false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && customAnswer.trim()) {
      handleCustomSubmit()
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Agent needs clarification
            </h3>
            <button
              onClick={onCancel}
              className="text-gray-400 hover:text-gray-600 text-xl font-bold"
            >
              ×
            </button>
          </div>
          
          <div className="mb-6">
            <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4">
              <p className="text-blue-800 font-medium">Question:</p>
              <p className="text-blue-700 mt-1">{clarification.question}</p>
            </div>
          </div>

          <div className="space-y-3 mb-6">
            <p className="text-sm font-medium text-gray-700">Choose an option:</p>
            {clarification.preset_answers.map((answer, index) => (
              <button
                key={index}
                onClick={() => handlePresetClick(index, answer)}
                className="w-full text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <span className="font-mono text-sm text-gray-500 mr-3">[{index + 1}]</span>
                <span className="text-gray-900">{answer}</span>
              </button>
            ))}
          </div>

          <div className="border-t pt-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Or provide a custom answer:</p>
            <div className="flex gap-2">
              <input
                type="text"
                value={customAnswer}
                onChange={(e) => setCustomAnswer(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your custom answer..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleCustomSubmit}
                disabled={!customAnswer.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Submit
              </button>
            </div>
          </div>

          <div className="mt-4 text-xs text-gray-500">
            Request pending for {Math.floor(clarification.elapsed_seconds)}s
          </div>
        </div>
      </div>
    </div>
  )
}

interface ClarificationProviderProps {
  children: React.ReactNode
}

export function ClarificationProvider({ children }: ClarificationProviderProps) {
  const [currentClarification, setCurrentClarification] = useState<ClarificationData | null>(null)
  const [isChecking, setIsChecking] = useState(false)

  // Poll for pending clarifications
  useEffect(() => {
    const checkForClarifications = async () => {
      if (isChecking || currentClarification) return
      
      setIsChecking(true)
      try {
        const response = await fetch('/api/v1/clarification/pending')
        const data = await response.json()
        
        if (data.pending_clarifications && data.pending_clarifications.length > 0) {
          // Show the first pending clarification
          setCurrentClarification(data.pending_clarifications[0])
        }
      } catch (error) {
        console.error('Error checking for clarifications:', error)
      } finally {
        setIsChecking(false)
      }
    }

    // Check every 5 seconds when no clarification is active (reduced from 2 seconds)
    const interval = setInterval(checkForClarifications, 5000)
    
    return () => clearInterval(interval)
  }, [isChecking, currentClarification])

  const handleAnswer = async (answer: string, isPreset: boolean, presetIndex?: number) => {
    if (!currentClarification) return

    try {
      const response = await fetch(`/api/v1/clarification/answer/${currentClarification.clarification_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          clarification_id: currentClarification.clarification_id,
          answer,
          is_preset: isPreset,
          preset_index: presetIndex,
        }),
      })

      if (response.ok) {
        setCurrentClarification(null)
      } else {
        console.error('Failed to submit answer')
      }
    } catch (error) {
      console.error('Error submitting answer:', error)
    }
  }

  const handleCancel = () => {
    // For now, just close the modal
    // In a production app, you might want to send a cancellation signal
    setCurrentClarification(null)
  }

  return (
    <>
      {children}
      {currentClarification && (
        <ClarificationModal
          clarification={currentClarification}
          onAnswer={handleAnswer}
          onCancel={handleCancel}
        />
      )}
    </>
  )
}