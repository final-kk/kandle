import React, { useState, useCallback, useRef } from 'react';

interface ChatInputProps {
  onSubmit: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

/**
 * Chat Input Component
 * For text-based interactions with Qwen3
 */
export const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  disabled = false,
  placeholder = 'Type your message...',
  className = '',
}) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      if (message.trim() && !disabled) {
        onSubmit(message.trim());
        setMessage('');
      }
    },
    [message, disabled, onSubmit]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <form onSubmit={handleSubmit} className={`flex gap-2 ${className}`}>
      <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={2}
          className="input-cyber resize-none pr-12"
        />
        <div className="absolute bottom-2 right-2 text-xs text-gray-600">
          ⏎ Send
        </div>
      </div>
      <button
        type="submit"
        disabled={disabled || !message.trim()}
        className="btn-cyber-primary px-6"
      >
        {disabled ? (
          <span className="animate-pulse">●●●</span>
        ) : (
          '→'
        )}
      </button>
    </form>
  );
};

export default ChatInput;
