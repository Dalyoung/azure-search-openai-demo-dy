import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "현대카드 홈페이지와 고객센터 정보 알려줘",
        value: "현대카드 홈페이지와 고객센터 정보 알려줘"
    },
    { text: "현대카드 제로 에디션 상품들의 정보 (연회비 등) 정리해서 알려줘", value: "현대카드 제로 에디션 상품들의 정보 (연회비 등) 정리해서 알려줘" },
    {
        text: "현대카드에서 진행하는 이벤트 중 자동차 살때 할인받을 수 있는 이벤트는 뭐 있어?",
        value: "현대카드에서 진행하는 이벤트 중 자동차 살때 할인받을 수 있는 이벤트가 있어?"
    }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
