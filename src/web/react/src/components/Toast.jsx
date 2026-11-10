import { useState, useEffect } from 'react';
import { registerToast } from '../utils/toast';

export default function Toast() {
  const [msg, setMsg]     = useState('');
  const [visible, setVisible] = useState(false);
  let timer = null;

  useEffect(() => {
    registerToast((m) => {
      setMsg(m);
      setVisible(true);
      clearTimeout(timer);
      timer = setTimeout(() => setVisible(false), 3000);
    });
  }, []);

  return <div className={`toast${visible ? ' show' : ''}`}>{msg}</div>;
}
