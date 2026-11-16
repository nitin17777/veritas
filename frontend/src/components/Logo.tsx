import { Icon } from "./Icons";

export function Logo() {
  return (
    <div className="logo">
      <span className="logo-mark">
        <Icon name="check" size={15} />
      </span>
      <span>VERITAS</span>
    </div>
  );
}
